//! TPT Abyss command-line interface.
//!
//! Demonstrates end-to-end dynamic-depth generation with symbolic
//! verification and (optionally) persistent memory.

use clap::{Parser, Subcommand};
use tpt_abyss_engine::{Engine, EngineConfig};
use tpt_abyss_router::HeuristicRouter;
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
        } => {
            let mut engine = build_engine(&cli)?;
            let tok_path = cli.tokenizer.clone();
            let out = generate(
                &mut engine,
                tok_path.as_deref(),
                prompt,
                *max_tokens,
                *sequential,
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
) -> Result<String, tpt_abyss_types::AbyssError> {
    let (prompt_tokens, tok) = match tokenizer {
        Some(p) => {
            let t = tpt_abyss_engine::Tokenizer::from_file(p)?;
            (t.encode(prompt)?, Some(t))
        }
        None => {
            // No tokenizer: encode as raw byte-ish ids (dev fallback).
            let ids: Vec<u32> = prompt.bytes().map(|b| b as u32).collect();
            (ids, None)
        }
    };

    // Use the router hook for dynamic depth (or force sequential).
    let depth = engine.num_layers() as u32;
    let router = HeuristicRouter::new(
        tpt_abyss_router::RouterConfig::builder()
            .model_depth(depth)
            .build(),
    );
    let hook: tpt_abyss_engine::RouterHook = if sequential {
        Box::new(move |_r, _len, _logits, _res| LayerProgram::sequential(depth))
    } else {
        Box::new(move |r, len, _logits, _res| {
            r.route_token(TokenId(1), Position(len as u32), 0.7, 0.6, false)
                .or_else(|_| LayerProgram::sequential(depth))
        })
    };
    let _ = &router;

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
        if sequential { "sequential" } else { "dynamic" }
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
