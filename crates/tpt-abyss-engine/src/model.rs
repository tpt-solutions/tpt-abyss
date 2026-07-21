//! GGUF-backed Llama-family model loader and weights for the TPT Abyss engine.
//!
//! We read the GGUF container ourselves via `candle_core::quantized::gguf_file`
//! and dequantize the needed tensors to `f32`, giving the engine full control
//! over execution order (required for non-sequential `LayerProgram`s) and the
//! KV cache. This is intentionally a focused Llama-family loader; other
//! architectures can be added behind the same `ModelWeights` type.
//!
//! ## Lazy block materialization (Phase 7.1)
//!
//! Instead of eagerly dequantizing every block at load time (which doubles peak
//! memory — raw quantized bytes + dequantized f32 copies), we store the GGUF
//! metadata and an mmap-backed handle. Blocks are dequantized on first use and
//! cached in a `HashMap<usize, BlockWeights>` so subsequent accesses (including
//! repeated layers in a `LayerProgram`) hit the cache.

use candle_core::quantized::gguf_file::{Content, Value};
use candle_core::{DType, Device, Result, Tensor};
use memmap2::Mmap;
use std::collections::HashMap;
use std::io::{Cursor, Seek};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub const DEFAULT_NEGATIVE_100: f32 = -100.0;

/// Resolved model hyper-parameters from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub arch: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attn_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub context_length: usize,
    pub rope_dims: usize,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
}

/// A single transformer block's weights (dequantized to f32).
#[derive(Debug, Clone)]
pub struct BlockWeights {
    pub attn_norm: Tensor,
    pub ffn_norm: Tensor,
    pub attn_q: Tensor,
    pub attn_k: Tensor,
    pub attn_v: Tensor,
    pub attn_q_bias: Option<Tensor>,
    pub attn_k_bias: Option<Tensor>,
    pub attn_v_bias: Option<Tensor>,
    pub attn_output: Tensor,
    pub ffn_up: Tensor,
    pub ffn_gate: Tensor,
    pub ffn_down: Tensor,
}

/// Metadata for a single block needed to materialize its weights from the GGUF
/// file on demand. Each `GgufTensorMeta` stores the tensor name so we can look
/// it up in the GGUF content's tensor registry and dequantize it lazily.
#[derive(Debug, Clone)]
struct GgufTensorMeta {
    name: String,
}

/// Metadata for all weights in a single transformer block.
#[derive(Debug, Clone)]
struct BlockMeta {
    attn_norm: GgufTensorMeta,
    ffn_norm: GgufTensorMeta,
    attn_q: GgufTensorMeta,
    attn_k: GgufTensorMeta,
    attn_v: GgufTensorMeta,
    attn_q_bias: Option<GgufTensorMeta>,
    attn_k_bias: Option<GgufTensorMeta>,
    attn_v_bias: Option<GgufTensorMeta>,
    attn_output: GgufTensorMeta,
    ffn_up: GgufTensorMeta,
    ffn_gate: GgufTensorMeta,
    ffn_down: GgufTensorMeta,
}

/// Shared, mmap-backed handle to the GGUF file. Stored inside [`ModelWeights`]
/// and cloned (cheaply, via `Arc`) when handing out references to the prefetch
/// worker or other threads.
#[derive(Clone)]
pub struct GgufSource {
    /// The mmap region. Dropped when all clones are gone.
    mmap: Arc<Mmap>,
    /// Parsed GGUF metadata (tensor offsets, metadata map).
    pub(crate) content: Arc<Content>,
    /// Filesystem path to the GGUF file (for diagnostics / re-open).
    #[allow(dead_code)]
    path: PathBuf,
}

impl GgufSource {
    /// Memory-map a GGUF file and parse its header + tensor descriptors.
    fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref())
            .map_err(|e| candle_core::Error::Msg(format!("open gguf: {e}")))?;
        // SAFETY: the mmap is held open for the lifetime of `GgufSource`; we
        // never unmap it while the model is alive. `Content::read` only needs
        // the file header, not the full data.
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| candle_core::Error::Msg(format!("mmap gguf: {e}")))?
        };
        let mut cursor = Cursor::new(&*mmap);
        let content =
            Content::read(&mut cursor).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        Ok(Self {
            mmap: Arc::new(mmap),
            content: Arc::new(content),
            path: path.as_ref().to_path_buf(),
        })
    }

    /// Return the names of all tensors belonging to the given block index,
    /// used by the background prefetch worker to fault in mmap pages.
    pub fn block_tensor_names(&self, block_idx: usize) -> Vec<String> {
        let prefix = format!("blk.{block_idx}.");
        self.content
            .tensor_infos
            .keys()
            .filter(|k| k.starts_with(&prefix))
            .cloned()
            .collect()
    }

    /// Return a cursor into the mmap for reading tensor data.
    pub fn cursor(&self) -> Cursor<&[u8]> {
        Cursor::new(&*self.mmap)
    }
}

/// Full model weights. Blocks are lazily dequantized from the mmap-backed GGUF
/// on first access, then cached for subsequent use.
pub struct ModelWeights {
    pub cfg: ModelConfig,
    pub embeddings: Tensor, // (vocab, hidden) — materialized eagerly (needed immediately)
    blocks_meta: Vec<BlockMeta>,
    block_cache: Mutex<HashMap<usize, BlockWeights>>,
    pub final_norm: Tensor, // materialized eagerly
    pub lm_head: Tensor,    // materialized eagerly
    source: GgufSource,
}

impl std::fmt::Debug for ModelWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelWeights")
            .field("cfg", &self.cfg)
            .field("num_blocks", &self.blocks_meta.len())
            .field("cached_blocks", &self.block_cache.lock().unwrap().len())
            .finish()
    }
}

/// Synthetic model construction for tests. Builds a ModelWeights directly from
/// pre-built block tensors without a GGUF file. The source is a dummy mmap
/// that is never accessed (all blocks are pre-cached).
impl ModelWeights {
    pub fn from_blocks(
        cfg: ModelConfig,
        embeddings: Tensor,
        blocks: Vec<BlockWeights>,
        final_norm: Tensor,
        lm_head: Tensor,
        _device: &Device,
    ) -> Result<Self> {
        let num_layers = blocks.len();
        let mut block_cache = HashMap::new();
        let mut blocks_meta = Vec::with_capacity(num_layers);
        for (i, bw) in blocks.into_iter().enumerate() {
            block_cache.insert(i, bw);
            blocks_meta.push(BlockMeta {
                attn_norm: GgufTensorMeta {
                    name: format!("synthetic.{i}.attn_norm"),
                },
                ffn_norm: GgufTensorMeta {
                    name: format!("synthetic.{i}.ffn_norm"),
                },
                attn_q: GgufTensorMeta {
                    name: format!("synthetic.{i}.attn_q"),
                },
                attn_k: GgufTensorMeta {
                    name: format!("synthetic.{i}.attn_k"),
                },
                attn_v: GgufTensorMeta {
                    name: format!("synthetic.{i}.attn_v"),
                },
                attn_q_bias: None,
                attn_k_bias: None,
                attn_v_bias: None,
                attn_output: GgufTensorMeta {
                    name: format!("synthetic.{i}.attn_output"),
                },
                ffn_up: GgufTensorMeta {
                    name: format!("synthetic.{i}.ffn_up"),
                },
                ffn_gate: GgufTensorMeta {
                    name: format!("synthetic.{i}.ffn_gate"),
                },
                ffn_down: GgufTensorMeta {
                    name: format!("synthetic.{i}.ffn_down"),
                },
            });
        }
        // Dummy GGUF source — synthetic models never read from it.
        let dummy_path = std::env::temp_dir().join("__tpt_abyss_synth__.gguf");
        let source = GgufSource {
            mmap: Arc::new(unsafe {
                Mmap::map(
                    &std::fs::File::create(&dummy_path)
                        .map_err(|e| candle_core::Error::Msg(format!("dummy gguf: {e}")))?,
                )
                .map_err(|e| candle_core::Error::Msg(format!("dummy mmap: {e}")))?
            }),
            content: Arc::new(Content {
                magic: candle_core::quantized::gguf_file::VersionedMagic::GgufV3,
                metadata: Default::default(),
                tensor_infos: Default::default(),
                tensor_data_offset: 0,
            }),
            path: dummy_path,
        };
        Ok(Self {
            cfg,
            embeddings,
            blocks_meta,
            block_cache: Mutex::new(block_cache),
            final_norm,
            lm_head,
            source,
        })
    }
}

fn get_u64(meta: &std::collections::HashMap<String, Value>, key: &str) -> Option<u64> {
    match meta.get(key)? {
        Value::U64(v) => Some(*v),
        Value::I64(v) => Some(*v as u64),
        Value::U32(v) => Some(*v as u64),
        Value::I32(v) => Some(*v as u64),
        Value::U16(v) => Some(*v as u64),
        Value::I16(v) => Some(*v as u64),
        Value::U8(v) => Some(*v as u64),
        Value::I8(v) => Some(*v as u64),
        _ => None,
    }
}

fn get_f32(meta: &std::collections::HashMap<String, Value>, key: &str) -> Option<f32> {
    match meta.get(key)? {
        Value::F32(v) => Some(*v),
        Value::F64(v) => Some(*v as f32),
        _ => None,
    }
}

fn get_string(meta: &std::collections::HashMap<String, Value>, key: &str) -> Option<String> {
    match meta.get(key)? {
        Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

/// Look up a tensor name in the GGUF content, returning the first match from
/// `candidates` that exists in the tensor registry.
fn find_tensor_name<'a>(content: &Content, candidates: &'a [&str]) -> Option<&'a str> {
    for c in candidates {
        if content.tensor_infos.contains_key(*c) {
            return Some(c);
        }
    }
    None
}

impl ModelConfig {
    /// Extract configuration from GGUF metadata, supporting the `llama` and
    /// `mistral` architectures (both use the same block layout).
    pub fn from_metadata(meta: &std::collections::HashMap<String, Value>) -> Result<Self> {
        let arch = get_string(meta, "general.architecture").unwrap_or_else(|| "llama".to_string());
        let p = arch.as_str();
        let block_count = get_u64(meta, &format!("{p}.block_count"))
            .or_else(|| get_u64(meta, &format!("{p}.n_layer")))
            .ok_or_else(|| candle_core::Error::Msg("missing block_count".into()))?
            as usize;
        let hidden = get_u64(meta, &format!("{p}.embedding_length"))
            .or_else(|| get_u64(meta, &format!("{p}.n_embd")))
            .ok_or_else(|| candle_core::Error::Msg("missing embedding_length".into()))?
            as usize;
        let n_heads = get_u64(meta, &format!("{p}.attention.head_count"))
            .ok_or_else(|| candle_core::Error::Msg("missing head_count".into()))?
            as usize;
        let n_kv = get_u64(meta, &format!("{p}.attention.head_count_kv")).unwrap_or(n_heads as u64)
            as usize;
        let head_dim = get_u64(meta, &format!("{p}.attention.key_length"))
            .or_else(|| get_u64(meta, &format!("{p}.attention.value_length")))
            .map(|v| v as usize)
            .unwrap_or(hidden / n_heads);
        let ffn = get_u64(meta, &format!("{p}.feed_forward_length"))
            .or_else(|| get_u64(meta, &format!("{p}.ffn_embed_dim")))
            .ok_or_else(|| candle_core::Error::Msg("missing feed_forward_length".into()))?
            as usize;
        let eps = get_f32(meta, &format!("{p}.attention.layer_norm_rms_epsilon"))
            .or_else(|| get_f32(meta, &format!("{p}.layer_norm_eps")))
            .unwrap_or(1e-5);
        let ctx = get_u64(meta, &format!("{p}.context_length"))
            .or_else(|| get_u64(meta, &format!("{p}.max_position_embeddings")))
            .unwrap_or(2048) as usize;
        let vocab = get_u64(meta, "general.vocab_size")
            .map(|v| v as usize)
            .unwrap_or(0);
        let rope_dims =
            get_u64(meta, &format!("{p}.rope.dims")).unwrap_or(head_dim as u64) as usize;
        let rope_theta = get_f32(meta, &format!("{p}.rope.theta"))
            .or_else(|| get_f32(meta, &format!("{p}.rope.freq_base")))
            .unwrap_or(10_000.0);
        Ok(Self {
            arch,
            hidden_size: hidden,
            num_layers: block_count,
            num_attn_heads: n_heads,
            num_kv_heads: n_kv,
            head_dim,
            intermediate_size: ffn,
            rms_norm_eps: eps,
            vocab_size: vocab,
            context_length: ctx,
            rope_dims,
            rope_theta,
            tie_word_embeddings: false,
        })
    }
}

impl ModelWeights {
    /// Load a GGUF model using mmap-backed lazy block materialization.
    ///
    /// Only the embedding matrix, final norm, and lm_head are dequantized
    /// eagerly. Transformer blocks are dequantized on first access and cached.
    pub fn from_gguf_mmap<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let source = GgufSource::open(path.as_ref())?;
        let mut cfg = ModelConfig::from_metadata(&source.content.metadata)?;

        let mut cursor = source.cursor();
        // Eagerly materialize the lightweight non-block tensors.
        let embeddings = materialize_tensor(
            &source.content,
            &mut cursor,
            device,
            &["token_embd.weight", "model.embed_tokens.weight"],
        )?;

        if cfg.vocab_size == 0 {
            cfg.vocab_size = embeddings.dims()[0];
        }

        let final_norm = materialize_tensor(
            &source.content,
            &mut cursor,
            device,
            &["output_norm.weight", "model.norm.weight"],
        )?;

        let lm_head = match find_tensor_name(&source.content, &["output.weight"]) {
            Some(name) => {
                let qt = source.content.tensor(&mut cursor, name, device)?;
                qt.dequantize(device)?.to_dtype(DType::F32)?
            }
            None => {
                cfg.tie_word_embeddings = true;
                embeddings.clone()
            }
        };

        // Build per-block metadata (no dequantization yet).
        let mut blocks_meta = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let meta = BlockMeta {
                attn_norm: GgufTensorMeta {
                    name: find_tensor_name(
                        &source.content,
                        &[
                            &format!("blk.{i}.attn_norm.weight"),
                            &format!("model.layers.{i}.input_layernorm.weight"),
                        ],
                    )
                    .ok_or_else(|| candle_core::Error::Msg(format!("missing attn_norm blk {i}")))?
                    .to_string(),
                },
                ffn_norm: GgufTensorMeta {
                    name: find_tensor_name(
                        &source.content,
                        &[
                            &format!("blk.{i}.ffn_norm.weight"),
                            &format!("model.layers.{i}.post_attention_layernorm.weight"),
                        ],
                    )
                    .ok_or_else(|| candle_core::Error::Msg(format!("missing ffn_norm blk {i}")))?
                    .to_string(),
                },
                attn_q: GgufTensorMeta {
                    name: find_tensor_name(
                        &source.content,
                        &[
                            &format!("blk.{i}.attn_q.weight"),
                            &format!("model.layers.{i}.self_attn.q_proj.weight"),
                        ],
                    )
                    .ok_or_else(|| candle_core::Error::Msg(format!("missing attn_q blk {i}")))?
                    .to_string(),
                },
                attn_k: GgufTensorMeta {
                    name: find_tensor_name(
                        &source.content,
                        &[
                            &format!("blk.{i}.attn_k.weight"),
                            &format!("model.layers.{i}.self_attn.k_proj.weight"),
                        ],
                    )
                    .ok_or_else(|| candle_core::Error::Msg(format!("missing attn_k blk {i}")))?
                    .to_string(),
                },
                attn_v: GgufTensorMeta {
                    name: find_tensor_name(
                        &source.content,
                        &[
                            &format!("blk.{i}.attn_v.weight"),
                            &format!("model.layers.{i}.self_attn.v_proj.weight"),
                        ],
                    )
                    .ok_or_else(|| candle_core::Error::Msg(format!("missing attn_v blk {i}")))?
                    .to_string(),
                },
                attn_q_bias: find_tensor_name(
                    &source.content,
                    &[
                        &format!("blk.{i}.attn_q.bias"),
                        &format!("model.layers.{i}.self_attn.q_proj.bias"),
                    ],
                )
                .map(|n| GgufTensorMeta {
                    name: n.to_string(),
                }),
                attn_k_bias: find_tensor_name(
                    &source.content,
                    &[
                        &format!("blk.{i}.attn_k.bias"),
                        &format!("model.layers.{i}.self_attn.k_proj.bias"),
                    ],
                )
                .map(|n| GgufTensorMeta {
                    name: n.to_string(),
                }),
                attn_v_bias: find_tensor_name(
                    &source.content,
                    &[
                        &format!("blk.{i}.attn_v.bias"),
                        &format!("model.layers.{i}.self_attn.v_proj.bias"),
                    ],
                )
                .map(|n| GgufTensorMeta {
                    name: n.to_string(),
                }),
                attn_output: GgufTensorMeta {
                    name: find_tensor_name(
                        &source.content,
                        &[
                            &format!("blk.{i}.attn_output.weight"),
                            &format!("model.layers.{i}.self_attn.o_proj.weight"),
                        ],
                    )
                    .ok_or_else(|| candle_core::Error::Msg(format!("missing attn_output blk {i}")))?
                    .to_string(),
                },
                ffn_up: GgufTensorMeta {
                    name: find_tensor_name(
                        &source.content,
                        &[
                            &format!("blk.{i}.ffn_up.weight"),
                            &format!("model.layers.{i}.mlp.up_proj.weight"),
                        ],
                    )
                    .ok_or_else(|| candle_core::Error::Msg(format!("missing ffn_up blk {i}")))?
                    .to_string(),
                },
                ffn_gate: GgufTensorMeta {
                    name: find_tensor_name(
                        &source.content,
                        &[
                            &format!("blk.{i}.ffn_gate.weight"),
                            &format!("model.layers.{i}.mlp.gate_proj.weight"),
                        ],
                    )
                    .ok_or_else(|| candle_core::Error::Msg(format!("missing ffn_gate blk {i}")))?
                    .to_string(),
                },
                ffn_down: GgufTensorMeta {
                    name: find_tensor_name(
                        &source.content,
                        &[
                            &format!("blk.{i}.ffn_down.weight"),
                            &format!("model.layers.{i}.mlp.down_proj.weight"),
                        ],
                    )
                    .ok_or_else(|| candle_core::Error::Msg(format!("missing ffn_down blk {i}")))?
                    .to_string(),
                },
            };
            blocks_meta.push(meta);
        }

        Ok(Self {
            cfg,
            embeddings,
            blocks_meta,
            block_cache: Mutex::new(HashMap::new()),
            final_norm,
            lm_head,
            source,
        })
    }

    /// Legacy eager-loading path: dequantizes all blocks upfront. Kept for
    /// backward compatibility with existing tests and the `from_weights` path.
    pub fn from_gguf<R: Seek + std::io::Read>(
        content: &Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut cfg = ModelConfig::from_metadata(&content.metadata)?;

        let mut tensors: std::collections::BTreeMap<String, Tensor> =
            std::collections::BTreeMap::new();
        let names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        for name in names {
            let qt = content.tensor(reader, &name, device)?;
            let t = qt.dequantize(device)?.to_dtype(DType::F32)?;
            tensors.insert(name, t);
        }

        let embeddings = tensors
            .get("token_embd.weight")
            .or_else(|| tensors.get("model.embed_tokens.weight"))
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("missing token_embd.weight".into()))?;

        if cfg.vocab_size == 0 {
            cfg.vocab_size = embeddings.dims()[0];
        }

        let mut blocks_meta = Vec::with_capacity(cfg.num_layers);
        let mut block_cache = HashMap::new();

        for i in 0..cfg.num_layers {
            let q = tensors
                .get(&format!("blk.{i}.attn_q.weight"))
                .or_else(|| tensors.get(&format!("model.layers.{i}.self_attn.q_proj.weight")))
                .cloned()
                .ok_or_else(|| candle_core::Error::Msg(format!("missing attn_q blk {i}")))?;
            let k = tensors
                .get(&format!("blk.{i}.attn_k.weight"))
                .or_else(|| tensors.get(&format!("model.layers.{i}.self_attn.k_proj.weight")))
                .cloned()
                .ok_or_else(|| candle_core::Error::Msg(format!("missing attn_k blk {i}")))?;
            let v = tensors
                .get(&format!("blk.{i}.attn_v.weight"))
                .or_else(|| tensors.get(&format!("model.layers.{i}.self_attn.v_proj.weight")))
                .cloned()
                .ok_or_else(|| candle_core::Error::Msg(format!("missing attn_v blk {i}")))?;
            let qb = tensors
                .get(&format!("blk.{i}.attn_q.bias"))
                .or_else(|| tensors.get(&format!("model.layers.{i}.self_attn.q_proj.bias")))
                .cloned();
            let kb = tensors
                .get(&format!("blk.{i}.attn_k.bias"))
                .or_else(|| tensors.get(&format!("model.layers.{i}.self_attn.k_proj.bias")))
                .cloned();
            let vb = tensors
                .get(&format!("blk.{i}.attn_v.bias"))
                .or_else(|| tensors.get(&format!("model.layers.{i}.self_attn.v_proj.bias")))
                .cloned();
            let o = tensors
                .get(&format!("blk.{i}.attn_output.weight"))
                .or_else(|| tensors.get(&format!("model.layers.{i}.self_attn.o_proj.weight")))
                .cloned()
                .ok_or_else(|| candle_core::Error::Msg(format!("missing attn_output blk {i}")))?;
            let up = tensors
                .get(&format!("blk.{i}.ffn_up.weight"))
                .or_else(|| tensors.get(&format!("model.layers.{i}.mlp.up_proj.weight")))
                .cloned()
                .ok_or_else(|| candle_core::Error::Msg(format!("missing ffn_up blk {i}")))?;
            let gate = tensors
                .get(&format!("blk.{i}.ffn_gate.weight"))
                .or_else(|| tensors.get(&format!("model.layers.{i}.mlp.gate_proj.weight")))
                .cloned()
                .ok_or_else(|| candle_core::Error::Msg(format!("missing ffn_gate blk {i}")))?;
            let down = tensors
                .get(&format!("blk.{i}.ffn_down.weight"))
                .or_else(|| tensors.get(&format!("model.layers.{i}.mlp.down_proj.weight")))
                .cloned()
                .ok_or_else(|| candle_core::Error::Msg(format!("missing ffn_down blk {i}")))?;
            let attn_norm = tensors
                .get(&format!("blk.{i}.attn_norm.weight"))
                .or_else(|| tensors.get(&format!("model.layers.{i}.input_layernorm.weight")))
                .cloned()
                .ok_or_else(|| candle_core::Error::Msg(format!("missing attn_norm blk {i}")))?;
            let ffn_norm = tensors
                .get(&format!("blk.{i}.ffn_norm.weight"))
                .or_else(|| {
                    tensors.get(&format!("model.layers.{i}.post_attention_layernorm.weight"))
                })
                .cloned()
                .ok_or_else(|| candle_core::Error::Msg(format!("missing ffn_norm blk {i}")))?;

            let meta = BlockMeta {
                attn_norm: GgufTensorMeta {
                    name: format!("blk.{i}.attn_norm.weight"),
                },
                ffn_norm: GgufTensorMeta {
                    name: format!("blk.{i}.ffn_norm.weight"),
                },
                attn_q: GgufTensorMeta {
                    name: format!("blk.{i}.attn_q.weight"),
                },
                attn_k: GgufTensorMeta {
                    name: format!("blk.{i}.attn_k.weight"),
                },
                attn_v: GgufTensorMeta {
                    name: format!("blk.{i}.attn_v.weight"),
                },
                attn_q_bias: qb.as_ref().map(|_| GgufTensorMeta {
                    name: format!("blk.{i}.attn_q.bias"),
                }),
                attn_k_bias: kb.as_ref().map(|_| GgufTensorMeta {
                    name: format!("blk.{i}.attn_k.bias"),
                }),
                attn_v_bias: vb.as_ref().map(|_| GgufTensorMeta {
                    name: format!("blk.{i}.attn_v.bias"),
                }),
                attn_output: GgufTensorMeta {
                    name: format!("blk.{i}.attn_output.weight"),
                },
                ffn_up: GgufTensorMeta {
                    name: format!("blk.{i}.ffn_up.weight"),
                },
                ffn_gate: GgufTensorMeta {
                    name: format!("blk.{i}.ffn_gate.weight"),
                },
                ffn_down: GgufTensorMeta {
                    name: format!("blk.{i}.ffn_down.weight"),
                },
            };

            block_cache.insert(
                i,
                BlockWeights {
                    attn_norm,
                    ffn_norm,
                    attn_q: q,
                    attn_k: k,
                    attn_v: v,
                    attn_q_bias: qb,
                    attn_k_bias: kb,
                    attn_v_bias: vb,
                    attn_output: o,
                    ffn_up: up,
                    ffn_gate: gate,
                    ffn_down: down,
                },
            );
            blocks_meta.push(meta);
        }

        let final_norm = tensors
            .get("output_norm.weight")
            .or_else(|| tensors.get("model.norm.weight"))
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("missing output_norm.weight".into()))?;

        let lm_head = match tensors.get("output.weight") {
            Some(t) => t.clone(),
            None => {
                cfg.tie_word_embeddings = true;
                embeddings.clone()
            }
        };

        // Build a dummy source (not used in legacy path, but required by struct).
        // Safety: We create a zero-length mmap-backed placeholder.
        let dummy_path = std::env::temp_dir().join("__tpt_abyss_dummy__.gguf");
        let source = GgufSource {
            mmap: Arc::new(unsafe {
                Mmap::map(
                    &std::fs::File::create(&dummy_path)
                        .map_err(|e| candle_core::Error::Msg(format!("dummy gguf: {e}")))?,
                )
                .map_err(|e| candle_core::Error::Msg(format!("dummy mmap: {e}")))?
            }),
            content: Arc::new(Content {
                magic: candle_core::quantized::gguf_file::VersionedMagic::GgufV3,
                metadata: Default::default(),
                tensor_infos: Default::default(),
                tensor_data_offset: 0,
            }),
            path: dummy_path,
        };

        Ok(Self {
            cfg,
            embeddings,
            blocks_meta,
            block_cache: Mutex::new(block_cache),
            final_norm,
            lm_head,
            source,
        })
    }

    /// Materialize a single block from the GGUF file, caching the result.
    /// Thread-safe: concurrent calls for different block ids proceed in parallel;
    /// duplicate calls for the same id will serialize on the Mutex.
    pub fn materialize_block(&self, block_idx: usize, device: &Device) -> Result<BlockWeights> {
        // Fast path: already cached.
        {
            let cache = self.block_cache.lock().unwrap();
            if let Some(bw) = cache.get(&block_idx) {
                return Ok(bw.clone());
            }
        }
        // Slow path: dequantize from the mmap.
        let bw = self.dequantize_block(block_idx, device)?;
        self.block_cache
            .lock()
            .unwrap()
            .insert(block_idx, bw.clone());
        Ok(bw)
    }

    /// Reference a block, materializing it lazily if needed. Returns a clone
    /// of the cached `BlockWeights`.
    #[inline]
    pub fn block(&self, block_idx: usize, device: &Device) -> Result<BlockWeights> {
        self.materialize_block(block_idx, device)
    }

    pub fn num_layers(&self) -> usize {
        self.cfg.num_layers
    }

    /// Number of blocks currently materialized in the cache.
    pub fn cached_block_count(&self) -> usize {
        self.block_cache.lock().unwrap().len()
    }

    /// Informational: we dequantize to f32 for full execution control.
    pub fn weight_precision(&self) -> &'static str {
        "dequantized-f32"
    }

    /// Return a clone of the GGUF source (shared via Arc). Used by the
    /// prefetch worker to dequantize blocks on a background thread.
    pub fn gguf_source(&self) -> GgufSource {
        self.source.clone()
    }

    /// Dequantize a single block's tensors from the GGUF file.
    fn dequantize_block(&self, block_idx: usize, device: &Device) -> Result<BlockWeights> {
        let meta = &self.blocks_meta[block_idx];
        let mut cursor = self.source.cursor();
        let content = &self.source.content;

        let attn_norm = dequantize_meta(content, &mut cursor, device, &meta.attn_norm)?;
        let ffn_norm = dequantize_meta(content, &mut cursor, device, &meta.ffn_norm)?;
        let attn_q = dequantize_meta(content, &mut cursor, device, &meta.attn_q)?;
        let attn_k = dequantize_meta(content, &mut cursor, device, &meta.attn_k)?;
        let attn_v = dequantize_meta(content, &mut cursor, device, &meta.attn_v)?;
        let attn_q_bias = match &meta.attn_q_bias {
            Some(m) => Some(dequantize_meta(content, &mut cursor, device, m)?),
            None => None,
        };
        let attn_k_bias = match &meta.attn_k_bias {
            Some(m) => Some(dequantize_meta(content, &mut cursor, device, m)?),
            None => None,
        };
        let attn_v_bias = match &meta.attn_v_bias {
            Some(m) => Some(dequantize_meta(content, &mut cursor, device, m)?),
            None => None,
        };
        let attn_output = dequantize_meta(content, &mut cursor, device, &meta.attn_output)?;
        let ffn_up = dequantize_meta(content, &mut cursor, device, &meta.ffn_up)?;
        let ffn_gate = dequantize_meta(content, &mut cursor, device, &meta.ffn_gate)?;
        let ffn_down = dequantize_meta(content, &mut cursor, device, &meta.ffn_down)?;

        Ok(BlockWeights {
            attn_norm,
            ffn_norm,
            attn_q,
            attn_k,
            attn_v,
            attn_q_bias,
            attn_k_bias,
            attn_v_bias,
            attn_output,
            ffn_up,
            ffn_gate,
            ffn_down,
        })
    }
}

/// Dequantize a single tensor by its GGUF metadata name.
fn dequantize_meta(
    content: &Content,
    reader: &mut Cursor<&[u8]>,
    device: &Device,
    meta: &GgufTensorMeta,
) -> Result<Tensor> {
    let qt = content
        .tensor(reader, &meta.name, device)
        .map_err(|e| candle_core::Error::Msg(format!("tensor '{}': {}", meta.name, e)))?;
    qt.dequantize(device)?.to_dtype(DType::F32)
}

/// Helper: materialize a single tensor from the GGUF content by trying multiple
/// candidate names (supports both Llama and Qwen2 naming conventions).
fn materialize_tensor(
    content: &Content,
    reader: &mut Cursor<&[u8]>,
    device: &Device,
    candidates: &[&str],
) -> Result<Tensor> {
    let name = find_tensor_name(content, candidates)
        .ok_or_else(|| candle_core::Error::Msg(format!("missing tensor: {:?}", candidates)))?;
    let qt = content.tensor(reader, name, device)?;
    qt.dequantize(device)?.to_dtype(DType::F32)
}
