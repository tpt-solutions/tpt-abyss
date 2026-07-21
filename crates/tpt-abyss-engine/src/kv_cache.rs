//! Per-layer KV cache for non-sequential execution.
//!
//! Because a `LayerProgram` may run a layer multiple times (e.g. layer 3 twice),
//! each layer maintains its **own** KV cache that grows independently every time
//! that layer executes. This is the dynamic KV-cache handling required by the
//! architecture: cache length per layer tracks how many times that layer has
//! actually run, not the global token position.
//!
//! ## Phase 7.2 — Device-aware caches
//!
//! `KvCachePool` supports per-layer device placement via a `ResidencyPlan`.
//! GPU-resident layers get their KV cache on GPU; CPU-resident layers get
//! theirs on CPU. This avoids CPU<->GPU data movement during attention for
//! layers that are pinned to one device.

use crate::device_placement::ResidencyPlan;
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
        // k_step: (n_kv_heads, seq, head_dim)
        self.k = Tensor::cat(&[&self.k, k_step], 1)?;
        self.v = Tensor::cat(&[&self.v, v_step], 1)?;
        Ok(())
    }

    /// Drop the most recently appended step's K/V (used by dynamic-depth
    /// "extra compute" passes so a repeated layer attends to true history
    /// rather than its own just-written entry, then re-appends it).
    pub fn pop(&mut self) -> Result<()> {
        let len = self.k.dims()[1];
        if len == 0 {
            return Ok(());
        }
        self.k = self.k.narrow(1, 0, len - 1)?;
        self.v = self.v.narrow(1, 0, len - 1)?;
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

    /// The device this layer's cache resides on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Reset this layer's cache to empty (used when migrating the layer
    /// to a different device, invalidating stale KV tensors).
    pub fn clear(&mut self) {
        *self = Self::new(self.n_kv_heads, self.head_dim, &self.device);
    }
}

/// A collection of per-layer KV caches, indexed by 0-based layer id.
///
/// Each layer's cache lives on the device specified by the `ResidencyPlan`.
/// This avoids cross-device data movement during self-attention for layers
/// that are pinned to one device (e.g. GPU-resident layers).
pub struct KvCachePool {
    layers: Vec<LayerKvCache>,
}

impl KvCachePool {
    /// Create a pool where all layers share the same device.
    pub fn new(num_layers: usize, n_kv_heads: usize, head_dim: usize, device: &Device) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerKvCache::new(n_kv_heads, head_dim, device))
            .collect();
        Self { layers }
    }

    /// Create a pool with per-layer device placement from a `ResidencyPlan`.
    /// GPU-resident layers get KV caches on GPU; CPU-resident layers on CPU.
    pub fn new_with_plan(
        num_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        plan: &ResidencyPlan,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|i| {
                let device = plan.device(i).unwrap_or(Device::Cpu);
                LayerKvCache::new(n_kv_heads, head_dim, &device)
            })
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
