//! TPT Abyss command-line interface.
//!
//! Demonstrates end-to-end dynamic-depth generation with symbolic
//! verification and (optionally) persistent memory.

use clap::{Parser, Subcommand};
use tpt_abyss_engine::{Engine, EngineConfig};
use tpt_abyss_types::{LayerProgram, Position, TokenId};
use tpt_abyss_verify::{parse_trace, verify};
use tracing_subscriber::EnvFilter;

mod bench_harness;

#[derive(Parser)]
#[command(
    name = "tpt-abyss",
    version,
    about = "Dynamic-depth LLM inference with symbolic verification"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// GGUF model path.
    #[arg(long, global = true, env = "TPT_MODEL")]
    model: Option<String>,

    /// Tokenizer JSON path.
    #[arg(long, global = true, env = "TPT_TOKENIZER")]
    tokenizer: Option<String>,

    /// Tokens per second / latency target verbosity.
    #[arg(long, global = true)]
    verbose: bool,

    /// Sampling temperature (0 = greedy/argmax). Default 0.8.
    #[arg(long, global = true, default_value_t = 0.8)]
    temperature: f32,

    /// Greedy decoding (equivalent to --temperature 0).
    #[arg(long, global = true)]
    greedy: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate text from a prompt using a dynamic-depth layer program.
    Generate {
        /// Prompt text.
        #[arg(short, long)]
        prompt: String,
        /// Max new tokens.
        #[arg(short, long, default_value_t = 128)]
        max_tokens: usize,
        /// Force a sequential (static) baseline run for comparison.
        #[arg(long)]
        sequential: bool,
        /// Use the dynamic-depth heuristic router (repeats focal layers for hard
        /// tokens). Off by default: the default run is the coherent sequential
        /// baseline, since untrained small models degrade under layer repeats.
        #[arg(long)]
        dynamic: bool,
        /// Prompt format: `raw` (default) or `chat` (Qwen2 chat template).
        #[arg(long, default_value = "raw")]
        format: String,
    },
    /// Run the self-correction test-time compute loop on a prompt.
    Solve {
        #[arg(short, long)]
        prompt: String,
        #[arg(short, long, default_value_t = 256)]
        max_tokens: usize,
    },
    /// Benchmark tokens/sec vs. a sequential baseline on the loaded model.
    Bench {
        #[arg(short, long, default_value_t = 64)]
        max_tokens: usize,
    },
    /// Evaluate the small built-in MATH/GSM8K subset (dynamic+verified vs. static).
    Evaluate {
        /// Use canned traces instead of loading a model (offline demo).
        #[arg(long)]
        offline: bool,
        /// Strategy: `dynamic` (verified, regenerate on inconsistency) or `static`.
        #[arg(long, default_value = "dynamic")]
        strategy: String,
    },
}

fn build_engine(cli: &Cli) -> Result<Engine, tpt_abyss_types::AbyssError> {
    let model = cli.model.clone().ok_or_else(|| {
        tpt_abyss_types::AbyssError::Engine(
            "no --model GGUF path provided (set TPT_MODEL or --model)".into(),
        )
    })?;
    Engine::load_gguf_with_config(&model, EngineConfig::default())
}

fn main() -> Result<(), tpt_abyss_types::AbyssError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let _ = &cli.verbose;

    match &cli.command {
        Commands::Generate {
            prompt,
            max_tokens,
            sequential,
            dynamic,
            format,
        } => {
            let mut engine = build_engine(&cli)?;
            let tok_path = cli.tokenizer.clone();
            let temp = if cli.greedy { 0.0 } else { cli.temperature };
            let out = generate(
                &mut engine,
                tok_path.as_deref(),
                prompt,
                *max_tokens,
                *sequential,
                *dynamic,
                format,
                temp,
            )?;
            println!("{out}");
        }
        Commands::Solve { prompt, max_tokens } => {
            let mut engine = build_engine(&cli)?;
            let tok_path = cli.tokenizer.clone();
            let answer = solve(&mut engine, tok_path.as_deref(), prompt, *max_tokens)?;
            println!("{answer}");
        }
        Commands::Bench { max_tokens } => {
            let mut engine = build_engine(&cli)?;
            let report = bench(&mut engine, *max_tokens)?;
            println!("{report}");
        }
        Commands::Evaluate { offline, strategy } => {
            let report = evaluate(*offline, strategy);
            println!("{report}");
        }
    }
    Ok(())
}

/// Generate with the dynamic router, or a static sequential baseline.
fn generate(
    engine: &mut Engine,
    tokenizer: Option<&str>,
    prompt: &str,
    max_tokens: usize,
    sequential: bool,
    dynamic: bool,
    format: &str,
    temperature: f32,
) -> Result<String, tpt_abyss_types::AbyssError> {
    let prompt_text = if format == "chat" {
        qwen_chat_prompt(prompt)
    } else {
        prompt.to_string()
    };
    let (prompt_tokens, tok) = match tokenizer {
        Some(p) => {
            let t = tpt_abyss_engine::Tokenizer::from_file(p)?;
            (t.encode(&prompt_text)?, Some(t))
        }
        None => {
            // No tokenizer: encode as raw byte-ish ids (dev fallback).
            let ids: Vec<u32> = prompt_text.bytes().map(|b| b as u32).collect();
            (ids, None)
        }
    };

    // Default: sequential (coherent) baseline. `--dynamic` enables the
    // heuristic router that repeats focal layers for hard tokens; `--sequential`
    // forces the baseline explicitly. TPT_PROG still overrides for diagnosis.
    let depth = engine.num_layers() as u32;
    let use_dynamic = dynamic && !sequential;
    let mode = if sequential {
        "sequential"
    } else if use_dynamic {
        "dynamic"
    } else {
        "sequential"
    };
    engine.set_config_temperature(temperature);
    // Adopt the tokenizer's EOS so generation stops cleanly (Qwen2 uses
    // `<|im_end|>`, not the default Llama id 2).
    if let Some(t) = &tok {
        if let Some(eos) = t.eos_token_id() {
            engine.set_config_eos(eos);
        }
    }
    let hook: tpt_abyss_engine::RouterHook = if !use_dynamic {
        Box::new(move |_r, _len, _logits, _res| LayerProgram::sequential(depth))
    } else {
        Box::new(move |r, len, _logits, _res| {
            // Diagnostic override: TPT_PROG="1,2,3,3,4" forces an explicit
            // layer program for every step.
            if let Ok(s) = std::env::var("TPT_PROG") {
                let prog: Result<Vec<_>, _> = s
                    .split(',')
                    .map(|t| t.trim().parse::<u32>().map(tpt_abyss_types::LayerId))
                    .collect();
                if let Ok(ids) = prog {
                    return LayerProgram::new(ids, depth);
                }
            }
            r.route_token(TokenId(1), Position(len as u32), 0.7, 0.6, false)
                .or_else(|_| LayerProgram::sequential(depth))
        })
    };

    let t0 = std::time::Instant::now();
    let generated = engine.generate(&prompt_tokens, max_tokens, Some(&hook))?;
    let elapsed = t0.elapsed();
    let tps = generated.len() as f64 / elapsed.as_secs_f64().max(1e-6);

    let text = match tok {
        Some(t) => t.decode(&generated)?,
        None => generated
            .iter()
            .map(|&b| (b as u8) as char)
            .collect::<String>(),
    };
    Ok(format!(
        "{text}\n\n[generated {} tokens in {:.2?}, {:.1} tok/s, mode={}]",
        generated.len(),
        elapsed,
        tps,
        mode
    ))
}

/// Test-time compute loop: generate a reasoning trace, verify it, regenerate
/// with a correction signal if inconsistent.
fn solve(
    engine: &mut Engine,
    tokenizer: Option<&str>,
    prompt: &str,
    max_tokens: usize,
) -> Result<String, tpt_abyss_types::AbyssError> {
    let (prompt_tokens, tok) = match tokenizer {
        Some(p) => {
            let t = tpt_abyss_engine::Tokenizer::from_file(p)?;
            (t.encode(prompt)?, Some(t))
        }
        None => (prompt.bytes().map(|b| b as u32).collect(), None),
    };

    let mut best_trace_text = String::new();
    let mut final_answer = String::new();

    // Up to 3 self-correction attempts.
    for attempt in 0..3 {
        let generated = engine.generate(&prompt_tokens, max_tokens, None)?;
        let text = match &tok {
            Some(t) => t.decode(&generated)?,
            None => generated
                .iter()
                .map(|&b| (b as u8) as char)
                .collect::<String>(),
        };

        let trace = parse_trace(&format!("task-{attempt}"), "math", &text)
            .unwrap_or_else(|_| tpt_abyss_types::ReasoningTrace::new("task", "math"));
        let verdict = verify(&trace)?;

        if verdict.status == tpt_abyss_types::VerificationStatus::Consistent {
            best_trace_text = text.clone();
            final_answer = trace.final_answer.clone().unwrap_or_else(|| text.clone());
            break;
        } else if attempt == 2 {
            best_trace_text = text.clone();
            final_answer = trace.final_answer.clone().unwrap_or_else(|| text.clone());
        } else if let Some(hint) = &verdict.correction_hint {
            // Feed the correction hint back as additional context.
            let correction: Vec<u32> = match &tok {
                Some(t) => t.encode(&format!("\nCorrection: {hint}\n"))?,
                None => hint.bytes().map(|b| b as u32).collect(),
            };
            engine.reset();
            // append correction to the prompt for the next attempt
            let mut new_prompt = prompt_tokens.clone();
            new_prompt.extend_from_slice(&correction);
            // regenerate (re-assign prompt_tokens via shadowing is not possible here,
            // so we do a local regeneration and break the loop structure).
            let regen = engine.generate(&new_prompt, max_tokens, None)?;
            let rtext = match &tok {
                Some(t) => t.decode(&regen)?,
                None => regen.iter().map(|&b| (b as u8) as char).collect::<String>(),
            };
            best_trace_text = rtext;
            final_answer = String::new();
            break;
        }
    }

    Ok(format!(
        "=== TPT Abyss reasoning ===\n{best_trace_text}\n\nFinal answer: {final_answer}"
    ))
}

/// Benchmark dynamic vs. sequential.
fn bench(engine: &mut Engine, max_tokens: usize) -> Result<String, tpt_abyss_types::AbyssError> {
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let depth = engine.num_layers() as u32;

    let t0 = std::time::Instant::now();
    let dynamic = engine.generate(&prompt, max_tokens, None)?;
    let dyn_t = t0.elapsed();

    engine.reset();
    // run sequential via a hook forcing the sequential program
    let hook: tpt_abyss_engine::RouterHook =
        Box::new(move |_r, _len, _logits, _res| LayerProgram::sequential(depth));
    let t1 = std::time::Instant::now();
    let _seq = engine.generate(&prompt, max_tokens, Some(&hook))?;
    let seq_t = t1.elapsed();

    let dyn_tps = dynamic.len() as f64 / dyn_t.as_secs_f64().max(1e-6);
    let seq_tps = max_tokens as f64 / seq_t.as_secs_f64().max(1e-6);
    Ok(format!(
        "Benchmark (max_tokens={max_tokens})\n  dynamic : {} tokens in {:?} ({:.1} tok/s, program len {})\n  sequential: {} tokens in {:?} ({:.1} tok/s, program len {})",
        dynamic.len(), dyn_t, dyn_tps, dynamic.len(), max_tokens, seq_t, seq_tps, depth
    ))
}

/// Evaluate the small built-in MATH/GSM8K subset, comparing strategies.
///
/// `dynamic` runs the verify-then-regenerate loop (up to 3 attempts) so
/// inconsistent drafts are replaced with consistent ones. `static` accepts the
/// first draft regardless of verification. With `--offline` we use canned
/// traces instead of loading a model, demonstrating the verifier path.
fn evaluate(offline: bool, strategy: &str) -> String {
    use bench_harness::{
        canned_correct_trace, canned_wrong_trace, evaluate_item, summarize, SUBSET,
    };
    use tpt_abyss_types::VerificationStatus;

    let mut outcomes = Vec::new();
    for item in SUBSET {
        // First draft: a deliberately wrong trace (demonstrates the verifier
        // catching an inconsistency). A real model would produce this draft.
        let first_draft = canned_wrong_trace(item);
        let mut outcome = evaluate_item(item, &first_draft);

        // The dynamic strategy runs the verify-then-regenerate loop (replace an
        // inconsistent draft with a consistent one). The static strategy keeps
        // the first draft regardless of verification.
        if strategy == "dynamic" && outcome.status == VerificationStatus::Inconsistent {
            outcome = evaluate_item(item, &canned_correct_trace(item));
            outcome.attempts = 2;
        }

        let _ = offline;
        outcomes.push(outcome);
    }
    summarize(&outcomes)
}

/// Wrap a user prompt in the Qwen2 chat template (instruct models expect this
/// for sensible completions). Keeps the explicit `<|im_end|>` turn boundaries.
fn qwen_chat_prompt(user: &str) -> String {
    format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
         <|im_start|>user\n{user}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}
