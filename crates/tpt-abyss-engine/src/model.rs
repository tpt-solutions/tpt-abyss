//! GGUF-backed Llama-family model loader and weights for the TPT Abyss engine.
//!
//! We read the GGUF container ourselves via `candle_core::quantized::gguf_file`
//! and dequantize the needed tensors to `f32`, giving the engine full control
//! over execution order (required for non-sequential `LayerProgram`s) and the
//! KV cache. This is intentionally a focused Llama-family loader; other
//! architectures can be added behind the same `ModelWeights` type.

use candle_core::quantized::gguf_file::{Content, Value};
use candle_core::{DType, Device, Result, Tensor};
use std::io::Seek;

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
    pub attn_output: Tensor,
    pub ffn_up: Tensor,
    pub ffn_gate: Tensor,
    pub ffn_down: Tensor,
}

/// Full dequantized model weights.
pub struct ModelWeights {
    pub cfg: ModelConfig,
    pub embeddings: Tensor, // (vocab, hidden)
    pub blocks: Vec<BlockWeights>,
    pub final_norm: Tensor,
    pub lm_head: Tensor, // (vocab, hidden)
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
            .ok_or_else(|| candle_core::Error::Msg("missing vocab_size".into()))?;
        let rope_dims =
            get_u64(meta, &format!("{p}.rope.dims")).unwrap_or(head_dim as u64) as usize;
        let rope_theta = get_f32(meta, &format!("{p}.rope.theta")).unwrap_or(10_000.0);
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
    /// Load and dequantize all weights from a GGUF `Content`.
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

        let mut blocks = Vec::with_capacity(cfg.num_layers);
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
            blocks.push(BlockWeights {
                attn_norm,
                ffn_norm,
                attn_q: q,
                attn_k: k,
                attn_v: v,
                attn_output: o,
                ffn_up: up,
                ffn_gate: gate,
                ffn_down: down,
            });
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

        Ok(Self {
            cfg,
            embeddings,
            blocks,
            final_norm,
            lm_head,
        })
    }

    pub fn num_layers(&self) -> usize {
        self.cfg.num_layers
    }

    /// Informational: we dequantize everything to f32 for full execution control.
    pub fn weight_precision(&self) -> &'static str {
        "dequantized-f32"
    }
}
