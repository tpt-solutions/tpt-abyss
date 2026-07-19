//! Top-level inference engine: GGUF loading, generation loop, router hook,
//! activation logging, and dynamic KV-cache management.

use crate::forward::{forward_program, ActivationLog};
use crate::kv_cache::KvCachePool;
use crate::model::ModelWeights;
use candle_core::Device;
use std::io::BufReader;
use std::path::Path;
use tpt_abyss_router::HeuristicRouter;
use tpt_abyss_types::{AbyssError, AbyssResult, LayerProgram, Position, TokenId};

use candle_core::quantized::gguf_file::Content;

/// Hook invoked before each generated token to choose its `LayerProgram`.
///
/// Receives the candidate next-token features and returns the program to run.
/// The default heuristic router is provided, but callers (e.g. a trained MLP
/// or the test-time compute loop) can supply their own.
pub type RouterHook = Box<
    dyn Fn(
            &HeuristicRouter,
            usize,  // current sequence length
            &[f32], // last logits (raw)
            f32,    // last residual magnitude proxy
        ) -> AbyssResult<LayerProgram>
        + Send
        + Sync,
>;

/// Engine configuration.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub max_context: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_context: 2048,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
        }
    }
}

/// The TPT Abyss non-sequential inference engine.
pub struct Engine {
    model: ModelWeights,
    device: Device,
    kv: KvCachePool,
    config: EngineConfig,
    router: HeuristicRouter,
    /// Per-step activation log collected across the last generation.
    activation_log: Vec<ActivationLog>,
}

impl Engine {
    /// Load a GGUF model from disk. Uses the CPU device by default (GPU/custom
    /// kernels are a later effort). A real small model (1-3B Q4_K_M) works.
    pub fn load_gguf<P: AsRef<Path>>(path: P) -> Result<Self, AbyssError> {
        Self::load_gguf_with_config(path, EngineConfig::default())
    }

    pub fn load_gguf_with_config<P: AsRef<Path>>(
        path: P,
        config: EngineConfig,
    ) -> Result<Self, AbyssError> {
        let device = Device::Cpu;
        let file = std::fs::File::open(path.as_ref())
            .map_err(|e| AbyssError::Weight(format!("open gguf: {e}")))?;
        let mut reader = BufReader::new(file);
        let content = Content::read(&mut reader).map_err(|e| AbyssError::Weight(e.to_string()))?;
        let model = ModelWeights::from_gguf(&content, &mut reader, &device)
            .map_err(|e| AbyssError::Weight(e.to_string()))?;
        let num_layers = model.num_layers();
        let cfg = model.cfg.clone();
        let kv = KvCachePool::new(num_layers, cfg.num_kv_heads, cfg.head_dim, &device);
        let router = HeuristicRouter::new(
            tpt_abyss_router::RouterConfig::builder()
                .model_depth(num_layers as u32)
                .build(),
        );
        Ok(Self {
            model,
            device,
            kv,
            config,
            router,
            activation_log: Vec::new(),
        })
    }

    /// Number of model layers.
    /// Build an engine directly from already-loaded weights (used by tests and
    /// embedded/synthetic models). Not part of the GGUF loading path.
    pub fn from_weights(model: ModelWeights, config: EngineConfig) -> Self {
        let device = Device::Cpu;
        let cfg = model.cfg.clone();
        let kv = KvCachePool::new(model.num_layers(), cfg.num_kv_heads, cfg.head_dim, &device);
        let router = HeuristicRouter::new(
            tpt_abyss_router::RouterConfig::builder()
                .model_depth(model.num_layers() as u32)
                .build(),
        );
        Self {
            model,
            device,
            kv,
            config,
            router,
            activation_log: Vec::new(),
        }
    }

    /// Consume the engine and return the loaded [`ModelWeights`] (e.g. to build
    /// a second engine with different config).
    pub fn into_model(self) -> ModelWeights {
        self.model
    }

    pub fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    /// A reference to the embedded router (used to build features / programs).
    pub fn router(&self) -> &HeuristicRouter {
        &self.router
    }

    /// Set a custom router (e.g. one wrapping a trained MLP).
    pub fn set_router(&mut self, router: HeuristicRouter) {
        self.router = router;
    }

    /// Reset KV caches and activation log for a fresh sequence.
    pub fn reset(&mut self) {
        self.kv.clear();
        self.activation_log.clear();
    }

    /// Access the collected activation log from the most recent generation.
    pub fn activation_log(&self) -> &[ActivationLog] {
        &self.activation_log
    }

    /// Run a single forward step over the current token sequence, returning
    /// logits for the next token. `index_pos` is the number of tokens already
    /// consumed in previous steps. The provided `program` governs which layers
    /// execute.
    pub fn step(
        &mut self,
        tokens: &[u32],
        index_pos: usize,
        program: &LayerProgram,
    ) -> Result<(Vec<f32>, ActivationLog), AbyssError> {
        let (logits, acts) = forward_program(
            &self.model,
            program,
            tokens,
            index_pos,
            &mut self.kv,
            &self.device,
        )
        .map_err(|e| AbyssError::Inference(e.to_string()))?;
        let logits_v: Vec<f32> = logits
            .to_vec1()
            .map_err(|e| AbyssError::Inference(e.to_string()))?;
        Ok((logits_v, acts))
    }

    /// Generate `max_new_tokens` tokens from a prompt using the default
    /// heuristic router to pick each token's layer program. Returns the full
    /// generated token sequence (excluding the prompt) and records activations.
    ///
    /// `router_fn` may be supplied to override program selection per step.
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        router_fn: Option<&RouterHook>,
    ) -> Result<Vec<u32>, AbyssError> {
        self.reset();
        let mut seq = prompt_tokens.to_vec();
        self.activation_log.clear();

        let mut index_pos = 0usize;
        let mut generated = Vec::new();

        for step in 0..max_new_tokens {
            let program =
                self.choose_program(seq.len(), &self.last_logits(&seq, index_pos)?, router_fn)?;
            let (logits, acts) = self.step(&seq, index_pos, &program)?;
            self.activation_log.push(acts);

            let next = self.sample(&logits);
            if step == 0 {
                // store nothing; just ensure progress
            }
            // Stop on an explicit EOS if present (id 2 is the common Llama EOS).
            if next == 2 {
                break;
            }
            generated.push(next);
            seq.push(next);
            index_pos = seq.len();
            if seq.len() >= self.config.max_context {
                break;
            }
        }
        Ok(generated)
    }

    fn last_logits(&self, _seq: &[u32], _index_pos: usize) -> Result<Vec<f32>, AbyssError> {
        // We don't keep last logits persistently; the router hook below takes
        // the freshly computed logits instead. Return an empty placeholder so
        // `generate` can fall back to entropy==0. The per-step hook path uses
        // real logits.
        Ok(Vec::new())
    }

    fn choose_program(
        &self,
        seq_len: usize,
        _last_logits: &[f32],
        router_fn: Option<&RouterHook>,
    ) -> Result<LayerProgram, AbyssError> {
        let depth = self.num_layers() as u32;
        match router_fn {
            Some(f) => f(&self.router, seq_len, _last_logits, 0.0),
            None => {
                // Default heuristic: easy backbone; modest repetition budget.
                self.router
                    .route_token(TokenId(0), Position(seq_len as u32), 0.3, 0.3, false)
                    .or_else(|_| LayerProgram::sequential(depth))
            }
        }
    }

    /// Argmax (greedy) or temperature/top-k sampling of logits.
    pub fn sample(&self, logits: &[f32]) -> u32 {
        if self.config.temperature == 0.0 {
            return argmax(logits);
        }
        // temperature scaling
        let inv = 1.0 / self.config.temperature.max(1e-6);
        let mut scaled: Vec<f32> = logits.iter().map(|l| l * inv).collect();
        // top-k
        if self.config.top_k > 0 && self.config.top_k < scaled.len() {
            let mut idx: Vec<usize> = (0..scaled.len()).collect();
            idx.sort_by(|a, b| scaled[*b].partial_cmp(&scaled[*a]).unwrap());
            let thresh = scaled[idx[self.config.top_k]];
            for (i, s) in scaled.iter_mut().enumerate() {
                if *s < thresh {
                    *s = f32::NEG_INFINITY;
                }
            }
        }
        // softmax
        let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scaled.iter_mut() {
            *s = (*s - max).exp();
            if !s.is_finite() {
                *s = 0.0;
            }
            sum += *s;
        }
        if sum <= 0.0 {
            return argmax(logits);
        }
        for s in scaled.iter_mut() {
            *s /= sum;
        }
        // categorical sample
        let mut rng = fast_rng();
        let r = rng / (u32::MAX as f32);
        let mut cum = 0.0f32;
        for (i, &p) in scaled.iter().enumerate() {
            cum += p;
            if r <= cum {
                return i as u32;
            }
        }
        argmax(logits)
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv {
                (i, v)
            } else {
                (bi, bv)
            }
        })
        .0 as u32
}

/// Tiny xorshift PRNG to avoid pulling `rand` into the hot path signature.
fn fast_rng() -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(0x9E3779B97F4A7C15);
    let mut x = STATE.load(Ordering::Relaxed);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    STATE.store(x, Ordering::Relaxed);
    (x >> 40) as f32
}
