use crate::AbyssError;
use serde::{Deserialize, Serialize};

/// The kind of a reasoning step parsed from a chain-of-thought trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepKind {
    /// A natural-language statement of intermediate reasoning.
    Statement,
    /// An arithmetic or algebraic computation step. e.g. "12 * 3 = 36".
    Computation,
    /// A logical deduction / inference step.
    Deduction,
    /// A conclusion drawn from prior steps.
    Conclusion,
    /// A goal / problem restatement.
    Goal,
}

/// A single structured step of a reasoning trace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningStep {
    /// 0-based index within the trace.
    pub index: usize,
    pub kind: StepKind,
    /// The raw text of the step.
    pub text: String,
    /// For computation steps, the parsed left operand (if numeric).
    pub lhs: Option<f64>,
    /// For computation steps, the operator (`+`, `-`, `*`, `/`, `=`)..
    pub op: Option<String>,
    /// For computation steps, the parsed right operand (if numeric).
    pub rhs: Option<f64>,
    /// For computation steps, the claimed result.
    pub result: Option<f64>,
}

/// A fully parsed chain-of-thought reasoning trace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningTrace {
    /// Identifier of the task/prompt that produced the trace.
    pub task_id: String,
    /// Free-form task type hint (e.g. "math", "logic", "code").
    pub task_type: String,
    pub steps: Vec<ReasoningStep>,
    /// The final answer text the model produced.
    pub final_answer: Option<String>,
}

impl ReasoningTrace {
    pub fn new(task_id: impl Into<String>, task_type: impl Into<String>) -> Self {
        Self {
            task_id: task_id.into(),
            task_type: task_type.into(),
            steps: Vec::new(),
            final_answer: None,
        }
    }

    /// Append a step, assigning the next index automatically.
    pub fn push_step(&mut self, step: ReasoningStep) {
        self.steps.push(step);
    }

    /// Validate that step indices are well-formed and in range.
    pub fn validate(&self) -> Result<(), AbyssError> {
        for (i, s) in self.steps.iter().enumerate() {
            if s.index != i {
                return Err(AbyssError::MalformedTrace {
                    reason: format!("step index {} but expected {}", s.index, i),
                });
            }
        }
        Ok(())
    }
}

/// The status outcome of a verification pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationStatus {
    /// No contradiction or arithmetic error found.
    Consistent,
    /// At least one contradiction or arithmetic error found.
    Inconsistent,
    /// Not enough information to verify.
    Undetermined,
}

/// A single detected problem within a trace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Violation {
    /// Index of the offending step (if known).
    pub step_index: Option<usize>,
    pub kind: String,
    pub message: String,
}

/// The result of verifying a [`ReasoningTrace`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationResult {
    pub status: VerificationStatus,
    /// Confidence in the verdict, in `[0.0, 1.0]`.
    pub confidence: f64,
    pub violations: Vec<Violation>,
    /// Suggested correction signal (e.g. "recompute step 3") for the engine.
    pub correction_hint: Option<String>,
}

impl VerificationResult {
    pub fn consistent(confidence: f64) -> Self {
        Self {
            status: VerificationStatus::Consistent,
            confidence,
            violations: Vec::new(),
            correction_hint: None,
        }
    }

    pub fn inconsistent(confidence: f64, violations: Vec<Violation>, hint: Option<String>) -> Self {
        Self {
            status: VerificationStatus::Inconsistent,
            confidence,
            violations,
            correction_hint: hint,
        }
    }

    /// Whether the engine should regenerate given this result.
    pub fn requires_regeneration(&self, threshold: f64) -> bool {
        self.status == VerificationStatus::Inconsistent && self.confidence >= threshold
    }
}
