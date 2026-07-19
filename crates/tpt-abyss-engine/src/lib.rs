//! TPT Abyss inference engine.
//!
//! A non-sequential (dynamic-depth) LLM inference engine built on `candle`.
//! The headline feature is [`forward_program`]: it runs an arbitrary
//! [`LayerProgram`] (e.g. `[1,2,3,3,4,5,5,6]`) instead of the fixed
//! 1..=N layer order. Repeated layers accumulate their own KV cache, so
//! "dynamic depth" is implemented correctly.
//!
//! Public entry points:
//! - [`Engine`] — GGUF loading, generation, router hook, activation logging.
//! - [`forward_program`] — the core non-sequential forward pass.
//! - [`ModelWeights`] / [`KvCachePool`] — lower-level building blocks.

pub mod engine;
pub mod forward;
pub mod kv_cache;
pub mod model;
pub mod tokenizer;

pub mod synthetic;

pub use engine::{Engine, EngineConfig, RouterHook};

pub use forward::{forward_program, ActivationLog};
pub use kv_cache::{KvCachePool, LayerKvCache};
pub use model::{BlockWeights, ModelConfig, ModelWeights};
pub use tokenizer::Tokenizer;
