//! From-scratch, in-process symbolic reasoner/verifier for TPT Abyss.
//!
//! `tpt-abyss-verify` turns a free-text chain-of-thought into a structured
//! [`ReasoningTrace`], then checks it for:
//! - arithmetic errors (`12 * 3 = 36` style step verification),
//! - logical contradictions (a later step negating an earlier claim),
//! - goal/answer consistency.
//!
//! It exposes a single plain-Rust entry point [`verify`] with no network hop,
//! returning a [`VerificationResult`] (status + confidence + correction hint)
//! that the engine feeds back into a regeneration loop.

mod arithmetic;
mod logic;
mod parser;
mod score;

pub use arithmetic::ArithmeticChecker;
pub use logic::ContradictionDetector;
pub use parser::{parse_trace, ParseError, TraceParser};
pub use score::ConfidenceModel;

use tpt_abyss_types::{AbyssResult, ReasoningTrace, VerificationResult};

/// Verify a reasoning trace and produce a verdict.
///
/// This is the core neural-symbolic interface: `draft -> verify -> accept or
/// regenerate`. The call is fully in-process and intended to run in well under
/// 100 ms for typical traces.
pub fn verify(trace: &ReasoningTrace) -> AbyssResult<VerificationResult> {
    trace.validate()?;
    let mut violations = Vec::new();

    let arith = ArithmeticChecker::new();
    arith.check(trace, &mut violations);

    let mut logic = ContradictionDetector::new();
    logic.check(trace, &mut violations);

    let confidence = ConfidenceModel::score(trace, &violations);
    let status = if violations.is_empty() {
        tpt_abyss_types::VerificationStatus::Consistent
    } else {
        tpt_abyss_types::VerificationStatus::Inconsistent
    };

    let hint = if status == tpt_abyss_types::VerificationStatus::Inconsistent {
        violations
            .first()
            .and_then(|v| v.step_index)
            .map(|i| format!("recompute or revise step {}", i))
    } else {
        None
    };

    Ok(VerificationResult {
        status,
        confidence,
        violations,
        correction_hint: hint,
    })
}
