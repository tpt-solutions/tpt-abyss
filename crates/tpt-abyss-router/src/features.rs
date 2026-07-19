use tpt_abyss_types::{Position, TokenId};

/// Per-token feature vector handed to the router.
///
/// For `v0.1` (heuristic, no trained weights) we derive a small, fixed set of
/// interpretable features from the token and generation state. The feature
/// vector is normalized to roughly `[-1, 1]` so a future learned MLP can be
/// trained on the same representation without rescaling.
#[derive(Debug, Clone)]
pub struct RouterFeatures {
    /// Number of hidden features (constant once configured).
    pub dim: usize,
    /// The flattened, normalized feature values.
    pub values: Vec<f32>,
}

impl RouterFeatures {
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Builder that produces a [`RouterFeatures`] vector of fixed dimension.
///
/// The feature layout (all in `[-1, 1]` where applicable):
/// 0: normalized position-in-sequence (position / max_position)
/// 1: token-id rarity proxy (id_low / id_high - 0.5) * 2
/// 2: entropy proxy of the last logits (surprise; high => harder)
/// 3: residual magnitude proxy (0..1, high => deeper compute useful)
/// 4: repetition flag (-1 normal, +1 if token recently repeated)
/// 5: progress flag (1 - position/max_position; high near start)
#[derive(Debug, Clone)]
pub struct RouterFeaturesBuilder {
    max_position: f32,
    id_high: f32,
    dim: usize,
}

impl RouterFeaturesBuilder {
    pub fn new(max_position: u32, vocab_size: u32) -> Self {
        Self {
            max_position: (max_position as f32).max(1.0),
            id_high: (vocab_size as f32).max(1.0),
            dim: 6,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn build(
        &self,
        token: TokenId,
        position: Position,
        logit_entropy: f32,
        residual_magnitude: f32,
        recently_repeated: bool,
    ) -> RouterFeatures {
        let f_pos = (position.0 as f32 / self.max_position) * 2.0 - 1.0;
        let f_id = (token.0 as f32 / self.id_high) * 2.0 - 1.0;
        let f_entropy = logit_entropy.clamp(0.0, 1.0) * 2.0 - 1.0;
        let f_resid = residual_magnitude.clamp(0.0, 1.0) * 2.0 - 1.0;
        let f_rep = if recently_repeated { 1.0 } else { -1.0 };
        let f_prog = (1.0 - position.0 as f32 / self.max_position) * 2.0 - 1.0;
        RouterFeatures {
            dim: self.dim,
            values: vec![f_pos, f_id, f_entropy, f_resid, f_rep, f_prog],
        }
    }
}
