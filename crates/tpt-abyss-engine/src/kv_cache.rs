//! Per-layer KV cache for non-sequential execution.
//!
//! Because a `LayerProgram` may run a layer multiple times (e.g. layer 3 twice),
//! each layer maintains its **own** KV cache that grows independently every time
//! that layer executes. This is the dynamic KV-cache handling required by the
//! architecture: cache length per layer tracks how many times that layer has
//! actually run, not the global token position.

use candle_core::{Device, Result, Tensor};

/// KV cache for a single transformer block.
#[derive(Debug, Clone)]
pub struct LayerKvCache {
    /// (n_kv_heads, seq_len_so_far, head_dim)
    k: Tensor,
    /// (n_kv_heads, seq_len_so_far, head_dim)
    v: Tensor,
    n_kv_heads: usize,
    head_dim: usize,
    device: Device,
}

impl LayerKvCache {
    pub fn new(n_kv_heads: usize, head_dim: usize, device: &Device) -> Self {
        // Start empty: (n_kv_heads, 0, head_dim).
        let k = Tensor::zeros((n_kv_heads, 0, head_dim), candle_core::DType::F32, device).unwrap();
        let v = Tensor::zeros((n_kv_heads, 0, head_dim), candle_core::DType::F32, device).unwrap();
        Self {
            k,
            v,
            n_kv_heads,
            head_dim,
            device: device.clone(),
        }
    }

    /// Append one step's K/V for this layer, extending the cache.
    pub fn append(&mut self, k_step: &Tensor, v_step: &Tensor) -> Result<()> {
        // k_step: (n_kv_heads, 1, head_dim)
        self.k = Tensor::cat(&[&self.k, k_step], 1)?;
        self.v = Tensor::cat(&[&self.v, v_step], 1)?;
        Ok(())
    }

    /// The full K/V seen by this layer so far.
    pub fn kv(&self) -> (&Tensor, &Tensor) {
        (&self.k, &self.v)
    }

    pub fn len(&self) -> usize {
        self.k.dims()[1]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A collection of per-layer KV caches, indexed by 0-based layer id.
pub struct KvCachePool {
    layers: Vec<LayerKvCache>,
}

impl KvCachePool {
    pub fn new(num_layers: usize, n_kv_heads: usize, head_dim: usize, device: &Device) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerKvCache::new(n_kv_heads, head_dim, device))
            .collect();
        Self { layers }
    }

    pub fn layer_mut(&mut self, layer: usize) -> &mut LayerKvCache {
        &mut self.layers[layer]
    }

    pub fn layer(&self, layer: usize) -> &LayerKvCache {
        &self.layers[layer]
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Reset all caches (start of a fresh sequence).
    pub fn clear(&mut self) {
        for l in &mut self.layers {
            *l = LayerKvCache::new(l.n_kv_heads, l.head_dim, &l.device);
        }
    }
}
