use std::collections::HashMap;
use tpt_abyss_types::{ReasoningTrace, StepKind, Violation};

/// Detects logical contradictions between reasoning steps.
///
/// A simple, domain-light approach: it tracks numeric claims of the form
/// "X = v" extracted from statement text, and flags later steps that assert
/// a different value for the same variable. It also flags a final answer that
/// disagrees with an earlier computed numeric claim.
pub struct ContradictionDetector {
    /// variable name -> last claimed value
    claims: HashMap<String, f64>,
}

impl ContradictionDetector {
    pub fn new() -> Self {
        Self {
            claims: HashMap::new(),
        }
    }

    pub fn check(&mut self, trace: &ReasoningTrace, out: &mut Vec<Violation>) {
        // First pass: collect variable claims.
        for step in &trace.steps {
            if let Some((var, val)) = extract_claim(&step.text) {
                if let Some(&prev) = self.claims.get(&var) {
                    if (prev - val).abs() > 1e-6 {
                        out.push(Violation {
                            step_index: Some(step.index),
                            kind: "contradiction".to_string(),
                            message: format!(
                                "step {} contradicts earlier claim: {} = {} vs {}",
                                step.index, var, val, prev
                            ),
                        });
                    }
                }
                self.claims.insert(var, val);
            }

            if step.kind == StepKind::Conclusion || step.kind == StepKind::Goal {
                // conclusions referencing a number inconsistent with claims
                if let Some((var, val)) = extract_claim(&step.text) {
                    if let Some(&prev) = self.claims.get(&var) {
                        if (prev - val).abs() > 1e-6 {
                            out.push(Violation {
                                step_index: Some(step.index),
                                kind: "contradiction".to_string(),
                                message: format!(
                                    "step {} conclusion contradicts claim for {}",
                                    step.index, var
                                ),
                            });
                        }
                    }
                }
            }
        }
    }
}

/// Extract a `var = number` claim from text (e.g. "total = 36").
fn extract_claim(text: &str) -> Option<(String, f64)> {
    let lower = text.to_ascii_lowercase();
    if let Some(eq) = lower.find('=') {
        let var = lower[..eq].trim().to_string();
        let val = lower[eq + 1..].trim().split_whitespace().next()?;
        if let Ok(v) = val.parse::<f64>() {
            if !var.is_empty() && var.chars().all(|c| c.is_alphanumeric() || c == ' ') {
                return Some((var.replace(' ', "_"), v));
            }
        }
    }
    None
}
