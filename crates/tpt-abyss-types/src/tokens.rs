use serde::{Deserialize, Serialize};
use std::fmt;

/// A vocabulary token id (0-based index into the model's vocab).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TokenId(pub u32);

impl fmt::Display for TokenId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// A decoded token (string piece). Cheap to clone on the small scale used here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Token {
    pub id: TokenId,
    pub text: String,
}

impl Token {
    pub fn new(id: TokenId, text: impl Into<String>) -> Self {
        Self {
            id,
            text: text.into(),
        }
    }
}

/// A position in the generated sequence (0-based).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Position(pub u32);

impl Position {
    pub fn next(self) -> Position {
        Position(self.0 + 1)
    }
    pub fn saturating_sub(self, n: u32) -> Position {
        Position(self.0.saturating_sub(n))
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pos{}", self.0)
    }
}
