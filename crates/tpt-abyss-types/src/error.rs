use thiserror::Error;

/// The crate-wide result type.
pub type AbyssResult<T> = Result<T, AbyssError>;

/// Errors shared across the TPT Abyss stack.
#[derive(Debug, Error)]
pub enum AbyssError {
    #[error("invalid layer program: {reason}")]
    InvalidLayerProgram { reason: String },

    #[error("malformed reasoning trace: {reason}")]
    MalformedTrace { reason: String },

    #[error("verification failed: {0}")]
    Verification(String),

    #[error("engine error: {0}")]
    Engine(String),

    #[error("router error: {0}")]
    Router(String),

    #[error("memory error: {0}")]
    Memory(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("model weight error: {0}")]
    Weight(String),

    #[error("inference error: {0}")]
    Inference(String),
}
