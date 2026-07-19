//! Embedded persistent memory for TPT Abyss (implemented in `storage.rs`).
pub mod storage;

pub use storage::{
    cosine, trivial_embedding, CausalRecord, MemoryStore, QualitySample, TraceRecord,
};
