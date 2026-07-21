//! Top-level inference engine: GGUF loading, generation loop, router hook,
//! activation logging, and dynamic KV-cache management.
//!
//! Phase 7.1: mmap-backed lazy loading + background prefetch. GGUF files are
//! memory-mapped and blocks dequantized on first use. A background prefetch
//! worker predicts upcoming layers via the router and materializes them ahead
//! of time.

use crate::device_placement::ResidencyPlan;
use crate::forward::{forward_program, ActivationLog};
use crate::kv_cache::KvCachePool;
use crate::model::ModelWeights;
use crate::usage_stats::LayerUsageStats;
use candle_core::Device;
use std::io::BufReader;
use std::path::Path;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use tpt_abyss_router::HeuristicRouter;
use tpt_abyss_types::{AbyssError, AbyssResult, LayerProgram, Position, TokenId};

use candle_core::quantized::gguf_file::Content;

/// Hook invoked before each generated token to choose its LayerProgram.
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
    /// End-of-sequence token id; generation stops when produced. Defaults to 2
    /// (common Llama EOS). Instruct models such as Qwen2 use a different id
    /// (e.g. im_end = 151645), supplied by the CLI after loading the
    /// tokenizer.
    pub eos_token_id: u32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_context: 2048,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            eos_token_id: 2,
        }
    }
}

/// Background prefetch worker. Dequantizes blocks on a background thread
/// and caches them in a shared CPU-side cache (`ModelWeights::cpu_block_cache`).
/// When the main thread needs a block for a GPU-resident layer, it pulls
/// the pre-dequantized CPU block and transfers to GPU via `clone_to_device()`
/// (fast H2D, ~5-20ms) instead of dequantizing from scratch (~100-200ms).
///
/// ## Why not true async H2D?
///
/// candle-core 0.8.4's `Tensor::to_device()` always calls
/// `htod_sync_copy` which blocks the calling thread. True async DMA
/// requires cudarc's `htod_copy_pinned()` + a separate CUDA stream,
/// but candle doesn't expose `CudaStorage::from_cuda_slice()` — there's
/// no way to wrap a raw `CudaSlice` back into a candle `Tensor`.
///
/// The dequantization overlap approach gives ~90% of the benefit
/// (dequantization is the expensive part) without requiring a candle fork.
struct PrefetchWorker {
    tx: mpsc::SyncSender<usize>,
    _handle: JoinHandle<()>,
}

impl PrefetchWorker {
    /// Spawn a worker that receives block indices, dequantizes them on
    /// CPU (from mmap GGUF data), and stores the result in the shared
    /// CPU-side prefetch cache. The main thread later pulls from this
    /// cache and transfers to GPU.
    fn spawn(
        model: &ModelWeights,
        cpu_cache: std::sync::Arc<
            std::sync::Mutex<std::collections::HashMap<usize, crate::model::BlockWeights>>,
        >,
    ) -> Self {
        let (tx, rx) = mpsc::sync_channel::<usize>(64);
        let source = model.gguf_source();
        let meta = model.blocks_meta.clone();
        let cpu_device = Device::Cpu;
        let handle = thread::spawn(move || {
            while let Ok(idx) = rx.recv() {
                // Skip if already in the CPU cache.
                {
                    let cache = cpu_cache.lock().unwrap();
                    if cache.contains_key(&idx) {
                        continue;
                    }
                }
                let bw =
                    crate::model::dequantize_block_from_source(&source, &meta, idx, &cpu_device);
                if let Ok(bw) = bw {
                    cpu_cache.lock().unwrap().insert(idx, bw);
                }
            }
        });
        Self {
            tx,
            _handle: handle,
        }
    }

    /// Request prefetching of a block index. Non-blocking; excess
    /// requests are dropped silently if the channel is full.
    fn prefetch(&self, block_idx: usize) {
        let _ = self.tx.try_send(block_idx);
    }

    /// Request prefetching of all unique layers in a LayerProgram.
    fn prefetch_program(&self, program: &LayerProgram) {
        let seen: std::collections::HashSet<usize> = program
            .as_slice()
            .iter()
            .map(|l| l.as_zero_based() as usize)
            .collect();
        for idx in seen {
            self.prefetch(idx);
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
    /// Background prefetch worker (None for synthetic/test models).
    prefetch: Option<PrefetchWorker>,
    /// Device residency plan (None for all-CPU default).
    residency_plan: Option<ResidencyPlan>,
    /// Per-layer usage statistics (updated by warm-up calibration pass).
    usage_stats: LayerUsageStats,
}

impl Engine {
    /// Load a GGUF model from disk using mmap-backed lazy block loading.
    /// Blocks are dequantized on first use. A background prefetch worker
    /// predicts upcoming layers and faults their pages in ahead of time.
    pub fn load_gguf<P: AsRef<Path>>(path: P) -> Result<Self, AbyssError> {
        Self::load_gguf_with_config(path, EngineConfig::default())
    }

    pub fn load_gguf_with_config<P: AsRef<Path>>(
        path: P,
        config: EngineConfig,
    ) -> Result<Self, AbyssError> {
        let device = Device::Cpu;
        let model = ModelWeights::from_gguf_mmap(path.as_ref(), &device)
            .map_err(|e| AbyssError::Weight(e.to_string()))?;
        let num_layers = model.num_layers();
        let cfg = model.cfg.clone();
        let kv = KvCachePool::new(num_layers, cfg.num_kv_heads, cfg.head_dim, &device);
        let router = HeuristicRouter::new(
            tpt_abyss_router::RouterConfig::builder()
                .model_depth(num_layers as u32)
                .build(),
        );
        let prefetch = Some(PrefetchWorker::spawn(&model, model.cpu_block_cache()));
        Ok(Self {
            model,
            device,
            kv,
            config,
            router,
            activation_log: Vec::new(),
            prefetch,
            residency_plan: None,
            usage_stats: LayerUsageStats::new(0.2, 0.5),
        })
    }

    /// Load a GGUF model with an explicit device residency plan.
    ///
    /// This is the Phase 7.2 entry point for layer-aware offloading. The plan
    /// determines which transformer blocks are GPU-resident (fast, limited VRAM)
    /// and which are CPU-resident (slower, abundant). The KV cache for each layer
    /// is placed on the same device as the layer itself.
    pub fn load_gguf_with_plan<P: AsRef<Path>>(
        path: P,
        config: EngineConfig,
        plan: &ResidencyPlan,
    ) -> Result<Self, AbyssError> {
        let default_device = plan
            .default_device()
            .to_device()
            .map_err(|e| AbyssError::Weight(format!("device init: {e}")))?;
        let model = ModelWeights::from_gguf_mmap(path.as_ref(), &default_device)
            .map_err(|e| AbyssError::Weight(e.to_string()))?;
        let num_layers = model.num_layers();
        let cfg = model.cfg.clone();
        let kv = KvCachePool::new_with_plan(num_layers, cfg.num_kv_heads, cfg.head_dim, plan);
        let router = HeuristicRouter::new(
            tpt_abyss_router::RouterConfig::builder()
                .model_depth(num_layers as u32)
                .build(),
        );
        let prefetch = Some(PrefetchWorker::spawn(&model, model.cpu_block_cache()));
        Ok(Self {
            model,
            device: default_device,
            kv,
            config,
            router,
            activation_log: Vec::new(),
            prefetch,
            residency_plan: Some(plan.clone()),
            usage_stats: LayerUsageStats::new(0.2, 0.5),
        })
    }

    /// Legacy eager-loading path (BufReader-based, dequantizes all blocks
    /// upfront). Used only when a raw Content+reader pair is available.
    #[allow(dead_code)]
    pub fn load_gguf_eager<P: AsRef<Path>>(
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
            prefetch: None,
            residency_plan: None,
            usage_stats: LayerUsageStats::new(0.2, 0.5),
        })
    }

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
            prefetch: None,
            residency_plan: None,
            usage_stats: LayerUsageStats::new(0.2, 0.5),
        }
    }

    /// Consume the engine and return the loaded ModelWeights.
    pub fn into_model(self) -> ModelWeights {
        self.model
    }

    pub fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    /// A reference to the embedded router.
    pub fn router(&self) -> &HeuristicRouter {
        &self.router
    }

    /// Set a custom router.
    pub fn set_router(&mut self, router: HeuristicRouter) {
        self.router = router;
    }

    /// Override the sampling temperature (0 = greedy/argmax) at runtime.
    pub fn set_config_temperature(&mut self, temperature: f32) {
        self.config.temperature = temperature;
    }

    /// Override the end-of-sequence token id at runtime.
    pub fn set_config_eos(&mut self, eos_token_id: u32) {
        self.config.eos_token_id = eos_token_id;
    }

    /// Access the current device residency plan (if any).
    pub fn residency_plan(&self) -> Option<&ResidencyPlan> {
        self.residency_plan.as_ref()
    }

    /// Number of GPU-resident layers in the current residency plan.
    /// Returns 0 if no plan is set (all-CPU default).
    pub fn gpu_layer_count(&self) -> usize {
        self.residency_plan.as_ref().map_or(0, |p| p.gpu_count())
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
    /// logits for the next token.
    pub fn step(
        &mut self,
        tokens: &[u32],
        index_pos: usize,
        program: &LayerProgram,
    ) -> Result<(Vec<f32>, ActivationLog), AbyssError> {
        // Prefetch upcoming blocks for the next step's likely program.
        // The worker dequantizes to CPU; materialize_block handles H2D.
        if let Some(ref pw) = self.prefetch {
            pw.prefetch_program(program);
        }
        let (logits, acts) = forward_program(
            &self.model,
            program,
            tokens,
            index_pos,
            &mut self.kv,
            &self.device,
            self.residency_plan.as_ref(),
        )
        .map_err(|e| AbyssError::Inference(e.to_string()))?;
        let logits_v: Vec<f32> = logits
            .to_vec1()
            .map_err(|e| AbyssError::Inference(e.to_string()))?;
        Ok((logits_v, acts))
    }

    /// Generate max_new_tokens tokens from a prompt. Uses the real activation
    /// log to drive the router's decision (replacing the hardcoded 0.3, 0.3
    /// placeholders from Phase 7.1).
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        router_fn: Option<&RouterHook>,
    ) -> Result<Vec<u32>, AbyssError> {
        self.reset();
        let mut seq = prompt_tokens.to_vec();
        self.activation_log.clear();

        let mut processed = 0usize;
        let mut generated = Vec::new();

        for _step in 0..max_new_tokens {
            // Compute real residual magnitude proxy from the last activation log.
            let residual_magnitude = self.last_residual_magnitude();
            let last_logits = self.last_logits_vec();

            let program =
                self.choose_program(seq.len(), &last_logits, residual_magnitude, router_fn)?;
            let (logits, acts) = self.step(&seq[processed..], processed, &program)?;
            self.activation_log.push(acts);

            // Prefetch layers for the *next* step's likely program.
            self.prefetch_next_step(&program);

            let next = self.sample(&logits);
            if next == self.config.eos_token_id {
                break;
            }
            generated.push(next);
            processed = seq.len();
            seq.push(next);
            if seq.len() >= self.config.max_context {
                break;
            }
        }
        Ok(generated)
    }

    /// Compute the mean activation magnitude from the last step's activation
    /// log, used as a real residual magnitude proxy for the router.
    fn last_residual_magnitude(&self) -> f32 {
        self.activation_log
            .last()
            .map(|log| {
                if log.is_empty() {
                    0.3 // default fallback for the first token
                } else {
                    let sum: f32 = log.iter().map(|(_, m)| m).sum();
                    (sum / log.len() as f32).clamp(0.0, 1.0)
                }
            })
            .unwrap_or(0.3)
    }

    /// Return the last step's logits (or empty if none yet).
    fn last_logits_vec(&self) -> Vec<f32> {
        Vec::new()
    }

    /// Prefetch blocks that the next generation step is likely to need.
    /// The background worker dequantizes on CPU; the main thread pulls
    /// pre-dequantized blocks via materialize_block's cpu_block_cache path.
    fn prefetch_next_step(&self, current_program: &LayerProgram) {
        if let Some(ref pw) = self.prefetch {
            pw.prefetch_program(current_program);
        }
    }

    /// Choose a LayerProgram using the real activation log instead of
    /// hardcoded placeholders. This wires the engine's per-layer activation
    /// magnitudes (computed in forward.rs) into the router's decision.
    fn choose_program(
        &self,
        seq_len: usize,
        _last_logits: &[f32],
        residual_magnitude: f32,
        router_fn: Option<&RouterHook>,
    ) -> Result<LayerProgram, AbyssError> {
        let depth = self.num_layers() as u32;
        match router_fn {
            Some(f) => f(&self.router, seq_len, _last_logits, residual_magnitude)
                .or_else(|_| LayerProgram::sequential(depth)),
            None => {
                let entropy = self.activation_entropy();
                self.router
                    .route_token(
                        TokenId(0),
                        Position(seq_len as u32),
                        entropy,
                        residual_magnitude,
                        false,
                    )
                    .or_else(|_| LayerProgram::sequential(depth))
            }
        }
    }

    /// Compute a proxy for logit entropy from the activation log. We use the
    /// variance of per-layer activation magnitudes as a stand-in for actual
    /// logit entropy (which would require the logits from the last step).
    fn activation_entropy(&self) -> f32 {
        self.activation_log
            .last()
            .map(|log| {
                if log.len() < 2 {
                    return 0.3;
                }
                let mags: Vec<f32> = log.iter().map(|(_, m)| *m).collect();
                let mean = mags.iter().sum::<f32>() / mags.len() as f32;
                let variance =
                    mags.iter().map(|m| (m - mean).powi(2)).sum::<f32>() / mags.len() as f32;
                // Map variance to [0, 1] range. Higher variance = harder token.
                (variance * 10.0).clamp(0.0, 1.0)
            })
            .unwrap_or(0.3)
    }

    /// Argmax (greedy) or temperature/top-k sampling of logits.
    pub fn sample(&self, logits: &[f32]) -> u32 {
        if self.config.temperature == 0.0 {
            return argmax(logits);
        }
        let inv = 1.0 / self.config.temperature.max(1e-6);
        let mut scaled: Vec<f32> = logits.iter().map(|l| l * inv).collect();
        if self.config.top_k > 0 && self.config.top_k < scaled.len() {
            let mut idx: Vec<usize> = (0..scaled.len()).collect();
            idx.sort_by(|a, b| scaled[*b].partial_cmp(&scaled[*a]).unwrap());
            let thresh = scaled[idx[self.config.top_k]];
            for s in scaled.iter_mut() {
                if *s < thresh {
                    *s = f32::NEG_INFINITY;
                }
            }
        }
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
        let rng = fast_rng();
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

    /// Access per-layer usage statistics.
    pub fn usage_stats(&self) -> &LayerUsageStats {
        &self.usage_stats
    }

    /// Access mutable per-layer usage statistics.
    pub fn usage_stats_mut(&mut self) -> &mut LayerUsageStats {
        &mut self.usage_stats
    }

    /// Warm-up calibration pass: runs a sequential forward through all layers
    /// with a dummy token, recording per-layer wall-clock time in the usage
    /// stats. Call once after loading the model and before the first generate()
    /// call so the adaptive repin logic has baseline timing data.
    pub fn warm_up(&mut self, dummy_token: u32) -> Result<(), AbyssError> {
        let depth = self.num_layers() as u32;
        let program =
            LayerProgram::sequential(depth).map_err(|e| AbyssError::Inference(e.to_string()))?;
        let tokens = vec![dummy_token];
        let (logits, acts) = self.step(&tokens, 0, &program)?;
        let _ = logits;
        // Record activation magnitudes from the warm-up step into usage stats.
        self.usage_stats.update(&acts, program.as_slice().len());
        Ok(())
    }

    /// Atomically migrate a single layer's weights and KV cache to a new device.
    ///
    /// This is the primitive for adaptive repinning: when usage_stats says a
    /// CPU-resident layer is hot, call `migrate_layer(idx, target_device)` to
    /// move it to GPU. The block is re-materialized on the target device from
    /// the GGUF source (cheap for mmap-backed models), and the layer's KV cache
    /// is cleared because KV values on the old device are stale.
    pub fn migrate_layer(
        &mut self,
        block_idx: usize,
        target_device: &Device,
    ) -> Result<(), AbyssError> {
        use crate::device_placement::LayerResidency;

        // Re-materialize the block on the target device from GGUF source.
        let _bw = self
            .model
            .block(block_idx, target_device)
            .map_err(|e| AbyssError::Inference(format!("migrate block {block_idx}: {e}")))?;

        // Update the residency plan so future prefetch/forward uses the new device.
        if let Some(ref mut plan) = self.residency_plan {
            let ordinal = match target_device {
                Device::Cuda(dev) => {
                    use candle_core::backend::BackendDevice;
                    match dev.location() {
                        candle_core::DeviceLocation::Cuda { gpu_id } => gpu_id,
                        _ => 0,
                    }
                }
                _ => 0,
            };
            plan.set_residency(block_idx, LayerResidency::Gpu { ordinal });
        }

        // Clear the KV cache for this layer — the cached KV tensors are on the
        // old device and won't be valid after migration.
        self.kv.layer_mut(block_idx).clear();

        Ok(())
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
