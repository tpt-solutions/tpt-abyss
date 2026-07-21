//! Dynamic depth router for TPT Abyss.
//!
//! The router converts a per-token feature vector (e.g. a normalized token
//! embedding summary plus positional/state features) into a [`LayerProgram`]:
//! an ordered list of layer indices to execute for that token.
//!
//! For `v0.1` the router is **heuristic / rule-based** — no trained MLP
//! weights yet. Training-data generation (from a 32B teacher) is a later
//! effort. The heuristic is deliberately structured so it can later be
//! replaced by a small learned MLP without changing the public API.
//!
//! Design goals (per `tpt-abyss-router.txt`):
//! - tiny: no heavy external primitive crates at this size
//! - fast: `route_token` must run in well under 1 ms on CPU
//! - panic-free: all indexing is bounds-checked

mod features;
mod heuristic;
mod math;
mod telemetry;

pub use features::{RouterFeatures, RouterFeaturesBuilder};
pub use heuristic::{HeuristicRouter, RouterConfig, RouterConfigBuilder};
pub use math::{matvec, softmax};
pub use telemetry::LayerSelectionStats;
