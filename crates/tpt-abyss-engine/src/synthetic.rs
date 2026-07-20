//! Synthetic model construction for tests: builds a tiny random-weight Llama
//! model entirely in-memory (no GGUF file) so the non-sequential forward pass
//! can be validated quickly. Not used in production.

use crate::model::{BlockWeights, ModelConfig, ModelWeights};
use candle_core::{DType, Device, Result, Tensor};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Build a minimal random model with the given hyper-parameters.
#[allow(clippy::too_many_arguments)]
pub fn synthetic_model(
    num_layers: usize,
    hidden: usize,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    ffn: usize,
    vocab: usize,
    device: &Device,
) -> Result<ModelWeights> {
    let cfg = ModelConfig {
        arch: "llama".to_string(),
        hidden_size: hidden,
        num_layers,
        num_attn_heads: n_heads,
        num_kv_heads: n_kv,
        head_dim,
        intermediate_size: ffn,
        rms_norm_eps: 1e-5,
        vocab_size: vocab,
        context_length: 256,
        rope_dims: head_dim,
        rope_theta: 10_000.0,
        tie_word_embeddings: true,
    };

    let mut rng = StdRng::seed_from_u64(0xAB55_1555);

    let mut r = || -> f32 { rand::Rng::gen_range(&mut rng, -0.1..0.1) };

    let t = |dims: &[usize], device: &Device, r: &mut dyn FnMut() -> f32| -> Result<Tensor> {
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n).map(|_| r()).collect();
        Tensor::from_vec(data, dims, device)
    };

    let embeddings = t(&[vocab, hidden], device, &mut r)?;
    let final_norm = t(&[hidden], device, &mut r)?;
    let lm_head = embeddings.clone(); // tied

    let mut blocks = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        blocks.push(BlockWeights {
            attn_norm: t(&[hidden], device, &mut r)?,
            ffn_norm: t(&[hidden], device, &mut r)?,
            attn_q: t(&[n_heads * head_dim, hidden], device, &mut r)?,
            attn_k: t(&[n_kv * head_dim, hidden], device, &mut r)?,
            attn_v: t(&[n_kv * head_dim, hidden], device, &mut r)?,
            attn_output: t(&[hidden, n_heads * head_dim], device, &mut r)?,
            ffn_up: t(&[ffn, hidden], device, &mut r)?,
            ffn_gate: t(&[ffn, hidden], device, &mut r)?,
            ffn_down: t(&[hidden, ffn], device, &mut r)?,
        });
    }

    let _ = DType::F32;
    Ok(ModelWeights {
        cfg,
        embeddings,
        blocks,
        final_norm,
        lm_head,
    })
}
