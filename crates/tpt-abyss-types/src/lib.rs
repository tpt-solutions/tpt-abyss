//! Shared types for the TPT Abyss dynamic-depth inference stack.
//!
//! These types are crate-agnostic building blocks used across the router,
//! engine, verifier, memory, and CLI crates:
//!
//! - [`LayerProgram`]: the ordered list of layer indices to execute
//!   (e.g. `[1, 2, 3, 3, 4, 5, 5, 6]`) — the core of non-sequential
//!   inference.
//! - Token / position primitives.
//! - [`ReasoningTrace`] / [`VerificationResult`]: the neural-symbolic
//!   feedback-loop payloads.
//! - [`AbyssError`]: the crate-wide error type.

mod error;
mod layer_program;
mod reasoning;
mod tokens;

pub use error::{AbyssError, AbyssResult};
pub use layer_program::{LayerId, LayerProgram, LayerProgramBuilder};
pub use reasoning::{
    ReasoningStep, ReasoningTrace, StepKind, VerificationResult, VerificationStatus, Violation,
};
pub use tokens::{Position, Token, TokenId};

/// Current semantic version of the shared type wire format.
pub const TYPES_FORMAT_VERSION: u32 = 1;
