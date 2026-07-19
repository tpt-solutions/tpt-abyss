//! Minimal benchmark harness comparing dynamic-depth + symbolic verification
//! against a static sequential baseline.
//!
//! This uses a small built-in MATH/GSM8K-style subset (not the full 500/1319
//! problems) so it can run offline without downloading external datasets. Each
//! item has a prompt, a reference numeric answer, and an expected reasoning
//! answer string. We approximate "model output" via the engine's generation
//! when a model is available; otherwise we run the verifier against a set of
//! canned candidate traces to measure the verify/regenerate loop's effect.

use tpt_abyss_types::{VerificationResult, VerificationStatus};
use tpt_abyss_verify::{parse_trace, verify};

/// A single evaluation item from a small MATH/GSM8K-style subset.
pub struct EvalItem {
    pub id: &'static str,
    pub task_type: &'static str,
    /// The prompt text (kept as dataset metadata; not read by the harness).
    #[allow(dead_code)]
    pub prompt: &'static str,
    /// Reference answer (numeric, compared loosely).
    pub answer: f64,
}

/// Tiny offline subset (replace with MATH-500 / GSM8K slices for real runs).
pub const SUBSET: &[EvalItem] = &[
    EvalItem {
        id: "gsm-01",
        task_type: "math",
        prompt: "If 3x + 5 = 20, what is x?",
        answer: 5.0,
    },
    EvalItem {
        id: "gsm-02",
        task_type: "math",
        prompt: "A book costs $12. If you buy 3 books with a 10% discount, how much?",
        answer: 32.4,
    },
    EvalItem {
        id: "math-01",
        task_type: "math",
        prompt: "What is 12 * 3?",
        answer: 36.0,
    },
    EvalItem {
        id: "math-02",
        task_type: "math",
        prompt: "Compute 144 / 12.",
        answer: 12.0,
    },
    EvalItem {
        id: "math-03",
        task_type: "math",
        prompt: "What is 7 + 8 * 2?",
        answer: 23.0,
    },
];

/// The result of evaluating one item under a strategy.
#[derive(Debug, Clone)]
pub struct EvalOutcome {
    #[allow(dead_code)]
    pub id: &'static str,
    pub status: VerificationStatus,
    pub confidence: f64,
    /// Whether the model's claimed answer matched the reference.
    pub answer_correct: bool,
    /// How many regeneration attempts were made.
    pub attempts: usize,
    #[allow(dead_code)]
    pub verdict: VerificationResult,
}

/// Run a single item's candidate reasoning text through the verifier and
/// compare the claimed final answer to the reference.
///
/// `candidate_text` is the model's drafted reasoning (with an `answer: <num>`
/// line). Returns the verification outcome.
pub fn evaluate_item(item: &EvalItem, candidate_text: &str) -> EvalOutcome {
    let trace = parse_trace(item.id, item.task_type, candidate_text)
        .unwrap_or_else(|_| tpt_abyss_types::ReasoningTrace::new(item.id, item.task_type));
    let verdict =
        verify(&trace).unwrap_or_else(|_| tpt_abyss_types::VerificationResult::consistent(0.0));

    let claimed = trace
        .final_answer
        .as_ref()
        .and_then(|a| a.trim().parse::<f64>().ok());
    let answer_correct = match claimed {
        Some(v) => (v - item.answer).abs() < 1e-6,
        None => false,
    };

    EvalOutcome {
        id: item.id,
        status: verdict.status,
        confidence: verdict.confidence,
        answer_correct,
        attempts: 1,
        verdict,
    }
}

/// Aggregate a set of outcomes into an accuracy / consistency report.
pub fn summarize(outcomes: &[EvalOutcome]) -> String {
    if outcomes.is_empty() {
        return "No items evaluated.".into();
    }
    let n = outcomes.len() as f64;
    let consistent = outcomes
        .iter()
        .filter(|o| o.status == VerificationStatus::Consistent)
        .count();
    let correct = outcomes.iter().filter(|o| o.answer_correct).count();
    let avg_conf = outcomes.iter().map(|o| o.confidence).sum::<f64>() / n;
    format!(
        "Evaluated {} items\n  answer accuracy : {}/{} ({:.1}%)\n  consistent     : {}/{} ({:.1}%)\n  avg confidence : {:.2}",
        outcomes.len(),
        correct,
        outcomes.len(),
        100.0 * correct as f64 / n,
        consistent,
        outcomes.len(),
        100.0 * consistent as f64 / n,
        avg_conf,
    )
}

/// A canned, correct reasoning trace for an item (used when no model is loaded,
/// to demonstrate the verify path end-to-end).
pub fn canned_correct_trace(item: &EvalItem) -> String {
    match item.id {
        "gsm-01" => "3x + 5 = 20\n3x = 15\nx = 5\nanswer: 5".into(),
        "gsm-02" => "3 * 12 = 36\n36 * 0.9 = 32.4\nanswer: 32.4".into(),
        "math-01" => "12 * 3 = 36\nanswer: 36".into(),
        "math-02" => "144 / 12 = 12\nanswer: 12".into(),
        "math-03" => "8 * 2 = 16\n7 + 16 = 23\nanswer: 23".into(),
        _ => format!("answer: {}", item.answer),
    }
}

/// A canned, *incorrect* reasoning trace (used to demonstrate the
/// regenerate-on-inconsistency path).
pub fn canned_wrong_trace(item: &EvalItem) -> String {
    match item.id {
        "gsm-01" => "3x + 5 = 20\n3x = 20\nx = 6.66\nanswer: 6.66".into(),
        "math-01" => "12 * 3 = 30\nanswer: 30".into(),
        _ => format!("answer: {}", item.answer + 1.0),
    }
}
