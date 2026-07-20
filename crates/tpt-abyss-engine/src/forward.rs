//! Non-sequential forward pass for the Llama-family model.
//!
//! This module implements the core TPT Abyss innovation: given a [`LayerProgram`]
//! (an arbitrary ordered list of layer indices such as `[1,2,3,3,4,5,5,6]`), the
//! forward pass executes exactly those layers in that order. Each executed
//! layer reads/writes its own [`LayerKvCache`], so a layer that runs twice
//! accumulates K/V across both runs — implementing "dynamic depth" correctly.

use crate::kv_cache::{KvCachePool, LayerKvCache};
use crate::model::ModelWeights;
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::rotary_emb;
use tpt_abyss_types::{LayerId, LayerProgram};

/// Per-layer activation magnitudes recorded during a forward pass (for
/// router-training data collection).
pub type ActivationLog = Vec<(LayerId, f32)>;

fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    // x: (b, seq, hidden)
    let x_dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
    let denom_term = variance.sqrt()?;
    // Build eps as a scalar-shaped tensor that broadcasts against [b, seq, 1].
    let eps_t = Tensor::new(&[eps], x.device())?.broadcast_as(denom_term.shape())?;
    let denominator = (&denom_term + &eps_t)?;
    let x = x.broadcast_div(&denominator)?;
    let x = x.to_dtype(x_dtype)?;
    let w = weight.broadcast_as(x.shape())?;
    x.broadcast_mul(&w)
}

fn silu(x: &Tensor) -> Result<Tensor> {
    // SiLU(x) = x * sigmoid(x).
    let sig = candle_nn::ops::sigmoid(x)?;
    x.broadcast_mul(&sig)
}

fn repeat_kv(x: &Tensor, repeat: usize) -> Result<Tensor> {
    // x: (b, n_kv, seq, hd)
    if repeat == 1 {
        return Ok(x.clone());
    }
    let (b, nkv, seq, hd) = x.dims4()?;
    let x = x.unsqueeze(2)?; // (b, nkv, 1, seq, hd)
    let x = x.broadcast_as((b, nkv, repeat, seq, hd))?;
    let x = x.reshape((b, nkv * repeat, seq, hd))?;
    Ok(x)
}

fn causal_mask(seq: usize, total: usize, device: &Device) -> Result<Tensor> {
    // allow q (0..seq) to attend keys (0..total) where key_index <= total - seq + q.
    let mut mask = vec![0.0f32; seq * total];
    for q in 0..seq {
        for k in 0..total {
            mask[q * total + k] = if k <= total - seq + q { 0.0 } else { -1e9 };
        }
    }
    Tensor::from_slice(&mask, (seq, total), device)
}

/// Run a single block (attention + MLP) on `x` for the given 0-based layer.
fn run_block(
    w: &crate::model::BlockWeights,
    cfg: &crate::model::ModelConfig,
    x: &Tensor,
    index_pos: usize,
    cache: &mut LayerKvCache,
    activations: &mut Vec<(LayerId, f32)>,
    layer_id: LayerId,
) -> Result<Tensor> {
    let (b, seq, hidden) = x.dims3()?;
    let n_heads = cfg.num_attn_heads;
    let n_kv = cfg.num_kv_heads;
    let head_dim = cfg.head_dim;
    let rope_dims = cfg.rope_dims;
    let device = x.device();

    // --- self attention ---
    let residual = x.clone();
    // Flatten (b, seq, hidden) -> (b*seq, hidden) for the linear projections so
    // the matmul is a plain 2D matrix multiply (robust across candle versions).
    let x2 = x.reshape((b * seq, hidden))?;
    let xn = rms_norm(&x2, &w.attn_norm, cfg.rms_norm_eps)?;
    let q = xn.matmul(&w.attn_q.t()?)?; // (b*seq, n_heads*head_dim)
    let k = xn.matmul(&w.attn_k.t()?)?; // (b*seq, n_kv*head_dim)
    let v = xn.matmul(&w.attn_v.t()?)?;

    let q = q.reshape((b, seq, n_heads, head_dim))?.transpose(1, 2)?;
    let k = k.reshape((b, seq, n_kv, head_dim))?.transpose(1, 2)?;
    let v = v.reshape((b, seq, n_kv, head_dim))?.transpose(1, 2)?;
    // (b, nh, seq, head_dim)

    // RoPE on q and k (rotate first rope_dims dims).
    let cos_sin = build_rope(seq, rope_dims, index_pos, cfg.rope_theta, device)?;
    let (cos, sin) = cos_sin;
    let q = rotary_emb::rope_i_slow(&q, &cos, &sin)?;
    let k = rotary_emb::rope_i_slow(&k, &cos, &sin)?;

    // append this step's k/v to the layer's own cache (drop the batch dim;
    // the cache stores per-layer (n_kv, seq, head_dim) state).
    cache.append(&k.squeeze(0)?, &v.squeeze(0)?)?;
    let (ck, cv) = cache.kv();
    let k_full = ck.unsqueeze(0)?; // (1, n_kv, total, hd)
    let v_full = cv.unsqueeze(0)?;
    let total = k_full.dims()[2];

    let repeat = n_heads / n_kv;
    let k_rep = repeat_kv(&k_full, repeat)?;
    let v_rep = repeat_kv(&v_full, repeat)?;

    let scale = 1.0 / (head_dim as f64).sqrt();
    let scores = q.matmul(&k_rep.t()?)?; // (1, nh, seq, total)
    let scale_t = Tensor::new(&[scale as f32], scores.device())?.broadcast_as(scores.shape())?;
    let scores = scores.broadcast_mul(&scale_t)?;
    let mask = causal_mask(seq, total, device)?;
    let scores = scores.broadcast_add(&mask.unsqueeze(0)?.unsqueeze(0)?)?;
    let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
    let ctx = attn.matmul(&v_rep)?; // (1, nh, seq, hd)
    let ctx = ctx.transpose(1, 2)?.reshape((b, seq, n_heads * head_dim))?;
    let attn_out = ctx
        .reshape((b * seq, n_heads * head_dim))?
        .matmul(&w.attn_output.t()?)?;
    let attn_out = attn_out.reshape((b, seq, hidden))?;
    let x = residual.broadcast_add(&attn_out)?;

    // --- mlp ---
    let residual2 = x.clone();
    let x2 = x.reshape((b * seq, hidden))?;
    let xn = rms_norm(&x2, &w.ffn_norm, cfg.rms_norm_eps)?;
    let gate = xn.matmul(&w.ffn_gate.t()?)?;
    let up = xn.matmul(&w.ffn_up.t()?)?;
    let x = silu(&gate)?.broadcast_mul(&up)?.matmul(&w.ffn_down.t()?)?;
    let x = x.reshape((b, seq, hidden))?;
    let x = residual2.broadcast_add(&x)?;

    let mag = x.abs()?.mean_all()?.to_scalar::<f32>()?;
    activations.push((layer_id, mag));
    Ok(x)
}

/// Build cos/sin tensors of shape (seq, rope_dims/2) for positions
/// `index_pos..index_pos+seq`.
fn build_rope(
    seq: usize,
    rope_dims: usize,
    index_pos: usize,
    theta: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = rope_dims / 2;
    let mut cos = vec![0.0f32; seq * half];
    let mut sin = vec![0.0f32; seq * half];
    for p in 0..seq {
        let pos = (index_pos + p) as f32;
        for i in 0..half {
            let angle = pos / theta.powf(2.0 * i as f32 / rope_dims as f32);
            cos[p * half + i] = angle.cos();
            sin[p * half + i] = angle.sin();
        }
    }
    let cos = Tensor::from_slice(&cos, (seq, half), device)?;
    let sin = Tensor::from_slice(&sin, (seq, half), device)?;
    Ok((cos, sin))
}

/// Execute a full forward pass following `program`, returning logits for the
/// last token plus the activation log.
pub fn forward_program(
    model: &ModelWeights,
    program: &LayerProgram,
    tokens: &[u32],
    index_pos: usize,
    kv: &mut KvCachePool,
    device: &Device,
) -> Result<(Tensor, ActivationLog)> {
    let cfg = &model.cfg;
    if tokens.is_empty() {
        candle_core::bail!("forward_program requires at least one token");
    }
    let seq = tokens.len();
    // embeddings: (vocab, hidden)
    // index_select expects indices shaped like (seq,) for selecting along dim 0.
    let tok_t = Tensor::from_slice(tokens, (seq,), device)?;
    let mut x = model.embeddings.index_select(&tok_t, 0)?; // (seq, hidden)

    x = x.unsqueeze(0)?; // (1, seq, hidden)
    x = x.to_dtype(DType::F32)?;

    let mut activations: ActivationLog = Vec::new();
    for &layer in program.as_slice() {
        let zero = layer.as_zero_based() as usize;
        let w = &model.blocks[zero];
        x = run_block(
            w,
            cfg,
            &x,
            index_pos,
            kv.layer_mut(zero),
            &mut activations,
            layer,
        )?;
    }

    let x_last = x.get(0)?; // (seq, hidden)
    let x_last = x_last.get(seq - 1)?; // (hidden,)
    let x_last = x_last.unsqueeze(0)?; // (1, hidden)
    let x_norm = rms_norm(&x_last, &model.final_norm, cfg.rms_norm_eps)?;
    let logits = x_norm.matmul(&model.lm_head.t()?)?; // (1, vocab)
    let logits = logits.squeeze(0)?; // (vocab,)
    Ok((logits, activations))
}
