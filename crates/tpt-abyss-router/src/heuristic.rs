use tpt_abyss_types::{AbyssResult, LayerProgram, LayerProgramBuilder};

use crate::features::{RouterFeatures, RouterFeaturesBuilder};

/// Configuration tuning the heuristic router's behavior.
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Total model depth (number of layers, 1-based max).
    pub model_depth: u32,
    /// Maximum number of times any single layer may be repeated.
    pub max_repeat: usize,
    /// Entropy above which a token is considered "hard" (gets extra compute).
    pub hard_entropy: f32,
    /// Residual-magnitude above which a token is considered "hard".
    pub hard_residual: f32,
    /// Cap on total program length to bound latency/compute.
    pub max_program_len: usize,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            model_depth: 32,
            max_repeat: 3,
            hard_entropy: 0.6,
            hard_residual: 0.6,
            max_program_len: 96,
        }
    }
}

impl RouterConfig {
    pub fn builder() -> RouterConfigBuilder {
        RouterConfigBuilder::default()
    }
}

/// Fluent builder for [`RouterConfig`].
#[derive(Debug, Clone, Default)]
pub struct RouterConfigBuilder {
    cfg: RouterConfig,
}

impl RouterConfigBuilder {
    pub fn model_depth(mut self, d: u32) -> Self {
        self.cfg.model_depth = d;
        self
    }
    pub fn max_repeat(mut self, r: usize) -> Self {
        self.cfg.max_repeat = r;
        self
    }
    pub fn hard_entropy(mut self, e: f32) -> Self {
        self.cfg.hard_entropy = e;
        self
    }
    pub fn hard_residual(mut self, r: f32) -> Self {
        self.cfg.hard_residual = r;
        self
    }
    pub fn max_program_len(mut self, n: usize) -> Self {
        self.cfg.max_program_len = n;
        self
    }
    pub fn build(self) -> RouterConfig {
        self.cfg
    }
}

/// A dependency-light, rule-based dynamic depth router.
///
/// The router emits a [`LayerProgram`] per token. Its current policy:
/// - Always run layers 1..=depth once (the backbone).
/// - If the token looks "hard" (high logit entropy or high residual
///   magnitude), repeat a single focal layer once to allocate a little extra
///   compute where it matters. Repeating is kept light (one extra pass of one
///   layer) so the program still tracks the model's trained distribution; the
///   capability to repeat arbitrary windows is fully supported by the engine
///   and is what later phases (trained routers) will exploit.
/// - Bound total length by `max_program_len`.
///
/// The policy is intentionally a pure function of [`RouterFeatures`] so it can
/// later be replaced by a trained small MLP with identical I/O.
#[derive(Debug, Clone)]
pub struct HeuristicRouter {
    cfg: RouterConfig,
    feature_builder: RouterFeaturesBuilder,
}

impl HeuristicRouter {
    pub fn new(cfg: RouterConfig) -> Self {
        let feature_builder = RouterFeaturesBuilder::new(2048, 32_000);
        Self {
            cfg,
            feature_builder,
        }
    }

    /// The model depth this router produces programs for.
    #[inline]
    pub fn model_depth(&self) -> u32 {
        self.cfg.model_depth
    }

    /// Build the feature vector for a token given its state signals.
    #[inline]
    pub fn features(
        &self,
        token: tpt_abyss_types::TokenId,
        position: tpt_abyss_types::Position,
        logit_entropy: f32,
        residual_magnitude: f32,
        recently_repeated: bool,
    ) -> RouterFeatures {
        self.feature_builder.build(
            token,
            position,
            logit_entropy,
            residual_magnitude,
            recently_repeated,
        )
    }

    /// Decide a layer program from already-computed features.
    pub fn route_features(&self, f: &RouterFeatures) -> AbyssResult<LayerProgram> {
        let depth = self.cfg.model_depth;
        // Hardness in [0,1]: max of normalized entropy and residual signals.
        let entropy = (f.values[2] + 1.0) / 2.0; // back to 0..1
        let residual = (f.values[3] + 1.0) / 2.0;
        let hardness = entropy.max(residual).clamp(0.0, 1.0);

        // How many extra single-layer passes to add for a hard token. Kept to
        // at most one so the program stays close to the trained distribution.
        let extra_passes =
            if hardness >= self.cfg.hard_entropy.max(self.cfg.hard_residual) || hardness >= 0.4 {
                1
            } else {
                0
            };

        let mut b = LayerProgramBuilder::with_depth(depth);
        // Backbone: sequential 1..=depth.
        for l in 1..=depth {
            b = b.layer(l);
        }
        // Repeat a single focal layer once for hard tokens, capped by
        // max_program_len. The focal layer is a late layer (near the top of the
        // stack), which gives a modest extra-compute signal without destabilising
        // the residual stream.
        if extra_passes > 0 && (depth as usize + 1) <= self.cfg.max_program_len {
            let focal = depth; // 1-based last layer
            for _ in 0..extra_passes {
                b = b.layer(focal);
            }
        }
        b.build()
    }

    /// Convenience: route directly from raw token/state signals.
    pub fn route_token(
        &self,
        token: tpt_abyss_types::TokenId,
        position: tpt_abyss_types::Position,
        logit_entropy: f32,
        residual_magnitude: f32,
        recently_repeated: bool,
    ) -> AbyssResult<LayerProgram> {
        let f = self.features(
            token,
            position,
            logit_entropy,
            residual_magnitude,
            recently_repeated,
        );
        self.route_features(&f)
    }
}
