use tpt_abyss_types::{ReasoningStep, ReasoningTrace, Violation};

/// Checks arithmetic computation steps for correctness.
pub struct ArithmeticChecker;

impl Default for ArithmeticChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl ArithmeticChecker {
    pub fn new() -> Self {
        Self
    }

    /// Append arithmetic violations into `out`.
    pub fn check(&self, trace: &ReasoningTrace, out: &mut Vec<Violation>) {
        for step in &trace.steps {
            if step.kind != tpt_abyss_types::StepKind::Computation {
                continue;
            }
            if let Some(v) = self.check_step(step) {
                out.push(v);
            }
        }
    }

    fn check_step(&self, step: &ReasoningStep) -> Option<Violation> {
        let (lhs, rhs, result) = match (step.lhs, step.rhs, step.result) {
            (Some(a), Some(b), Some(r)) => (a, b, r),
            _ => return None, // not parseable as a full equation; skip
        };
        let op = step.op.as_deref()?;
        let expected = match op {
            "+" => lhs + rhs,
            "-" => lhs - rhs,
            "*" => lhs * rhs,
            "/" => {
                if rhs == 0.0 {
                    return Some(Violation {
                        step_index: Some(step.index),
                        kind: "division_by_zero".to_string(),
                        message: format!("division by zero in step {}", step.index),
                    });
                }
                lhs / rhs
            }
            _ => return None,
        };
        // Allow tiny float tolerance.
        if (expected - result).abs() > 1e-6 {
            return Some(Violation {
                step_index: Some(step.index),
                kind: "arithmetic_error".to_string(),
                message: format!(
                    "step {}: {} {} {} = {} but claimed {}",
                    step.index, lhs, op, rhs, expected, result
                ),
            });
        }
        None
    }
}
