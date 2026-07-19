use tpt_abyss_types::{ReasoningTrace, VerificationStatus, Violation};

/// Produces a confidence score in `[0, 1]` for a verification verdict.
///
/// Confidence is higher when: the trace is longer/structured (more evidence),
/// no violations were found, and computation steps dominate verifiable claims.
/// When violations exist, confidence scales with how many steps were checked
/// vs. how many failed, and how severe the failure is.
pub struct ConfidenceModel;

impl ConfidenceModel {
    pub fn score(trace: &ReasoningTrace, violations: &[Violation]) -> f64 {
        if trace.steps.is_empty() {
            return 0.0;
        }
        let total = trace.steps.len() as f64;
        let computable = trace
            .steps
            .iter()
            .filter(|s| s.kind == tpt_abyss_types::StepKind::Computation)
            .count() as f64;

        if violations.is_empty() {
            // More structure => more evidence => higher confidence, capped.
            let structure = (total / 10.0).min(1.0);
            return 0.7 + 0.3 * structure;
        }

        // Some violations: lower confidence proportionally to failed share.
        let fail_share = (violations.len() as f64) / total.max(1.0);
        let severity = violations
            .iter()
            .filter(|v| v.kind == "division_by_zero" || v.kind == "contradiction")
            .count() as f64
            / violations.len().max(1) as f64;
        let base = 0.5 * (1.0 - fail_share) + 0.3 * (computable / total.max(1.0));
        (base * (1.0 - 0.5 * severity)).clamp(0.0, 1.0)
    }

    /// Convenience to reconstruct a status from violations (already done in
    /// [`crate::verify`], kept for API symmetry).
    pub fn status_for(violations: &[Violation]) -> VerificationStatus {
        if violations.is_empty() {
            VerificationStatus::Consistent
        } else {
            VerificationStatus::Inconsistent
        }
    }
}
