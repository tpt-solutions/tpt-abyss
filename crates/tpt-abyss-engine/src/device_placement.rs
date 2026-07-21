//! Device specification and per-layer residency planning (Phase 7.2).
//!
//! `DeviceSpec` describes a compute device (CPU, CUDA GPU 0, etc.).
//! `ResidencyPlan` maps each transformer block to a target device, controlling
//! which layers are GPU-resident vs CPU-resident. This is the core of
//! layer-aware offloading: only actively-used layers need to be on GPU.
//!
//! ## Design
//!
//! A `ResidencyPlan` is a fixed-length mapping from 0-based block index to
//! device. It is produced by one of:
//! - A static config (equivalent to llama.cpp's `--n-gpu-layers`)
//! - The adaptive residency algorithm (Phase 7.3) which uses router telemetry
//!   to pin hot layers to GPU and evict cold layers to CPU.
//!
//! The engine applies the plan when materializing blocks: GPU-resident blocks
//! are dequantized directly to GPU tensors, while CPU-resident blocks stay in
//! CPU memory. The `KvCachePool` uses the plan to decide whether each layer's
//! KV cache lives on GPU or CPU.

use candle_core::Device;

/// Describes a compute device.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DeviceSpec {
    /// CPU (always available, no special requirements).
    Cpu,
    /// CUDA GPU by ordinal (0 = first GPU).
    Cuda { ordinal: usize },
}

impl DeviceSpec {
    /// Create a candle `Device` from this specification.
    ///
    /// Returns `Err` if the requested device is not available (e.g. no CUDA
    /// support compiled in, or ordinal out of range).
    pub fn to_device(&self) -> Result<Device, String> {
        match self {
            DeviceSpec::Cpu => Ok(Device::Cpu),
            DeviceSpec::Cuda { ordinal } => {
                #[cfg(feature = "cuda")]
                {
                    use candle_core::backend::BackendDevice;
                    let dev = candle_core::CudaDevice::new(*ordinal)
                        .map_err(|e| format!("cuda device {ordinal}: {e}"))?;
                    Ok(Device::Cuda(dev))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = ordinal;
                    Err("CUDA support not compiled (enable the 'cuda' feature)".into())
                }
            }
        }
    }

    /// Default: CPU.
    pub fn default_cpu() -> Self {
        DeviceSpec::Cpu
    }

    /// Default: CUDA device 0 (if available).
    pub fn default_cuda() -> Self {
        DeviceSpec::Cuda { ordinal: 0 }
    }
}

impl std::fmt::Display for DeviceSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceSpec::Cpu => write!(f, "cpu"),
            DeviceSpec::Cuda { ordinal } => write!(f, "cuda:{ordinal}"),
        }
    }
}

/// The residency (device placement) of a single transformer block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerResidency {
    /// Block is dequantized and kept on CPU.
    Cpu,
    /// Block is dequantized and kept on a CUDA GPU.
    Gpu { ordinal: usize },
}

impl LayerResidency {
    /// Convert to a `DeviceSpec`.
    pub fn device_spec(&self) -> DeviceSpec {
        match self {
            LayerResidency::Cpu => DeviceSpec::Cpu,
            LayerResidency::Gpu { ordinal } => DeviceSpec::Cuda { ordinal: *ordinal },
        }
    }

    /// Convert to a candle `Device`.
    pub fn to_device(&self) -> Result<Device, String> {
        self.device_spec().to_device()
    }
}

/// A complete mapping from 0-based block index to device residency.
///
/// `ResidencyPlan` is the core scheduling output of the layer-aware offloading
/// system. It determines which transformer blocks live on GPU (fast, limited
/// VRAM) and which live in CPU RAM (slower, abundant).
#[derive(Debug, Clone)]
pub struct ResidencyPlan {
    /// Per-block residency. Index = 0-based block id.
    layers: Vec<LayerResidency>,
    /// The default device for blocks not explicitly mapped (e.g. embeddings,
    /// lm_head). Typically the device with the most VRAM.
    default_device: DeviceSpec,
}

impl ResidencyPlan {
    /// All-CPU plan (no GPU offloading). This is the v0.1 default.
    pub fn all_cpu(num_layers: usize) -> Self {
        Self {
            layers: vec![LayerResidency::Cpu; num_layers],
            default_device: DeviceSpec::Cpu,
        }
    }

    /// All-GPU plan (entire model on GPU). Equivalent to llama.cpp's
    /// `--n-gpu-layers` set to the full layer count.
    pub fn all_gpu(num_layers: usize, ordinal: usize) -> Self {
        Self {
            layers: vec![LayerResidency::Gpu { ordinal }; num_layers],
            default_device: DeviceSpec::Cuda { ordinal },
        }
    }

    /// Static split: first `gpu_layers` blocks on GPU, the rest on CPU.
    /// Equivalent to llama.cpp's `--n-gpu-layers N`.
    pub fn static_split(num_layers: usize, gpu_layers: usize, ordinal: usize) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            if i < gpu_layers {
                layers.push(LayerResidency::Gpu { ordinal });
            } else {
                layers.push(LayerResidency::Cpu);
            }
        }
        Self {
            layers,
            default_device: if gpu_layers > 0 {
                DeviceSpec::Cuda { ordinal }
            } else {
                DeviceSpec::Cpu
            },
        }
    }

    /// Residency for a specific block.
    pub fn residency(&self, block_idx: usize) -> &LayerResidency {
        &self.layers[block_idx]
    }

    /// Device for a specific block.
    pub fn device(&self, block_idx: usize) -> Result<Device, String> {
        self.layers[block_idx].to_device()
    }

    /// The default device (for non-block tensors like embeddings).
    pub fn default_device(&self) -> &DeviceSpec {
        &self.default_device
    }

    /// Number of blocks in the plan.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Number of GPU-resident layers.
    pub fn gpu_count(&self) -> usize {
        self.layers
            .iter()
            .filter(|r| matches!(r, LayerResidency::Gpu { .. }))
            .count()
    }

    /// Number of CPU-resident layers.
    pub fn cpu_count(&self) -> usize {
        self.layers.len() - self.gpu_count()
    }

    /// Replace a specific block's residency.
    pub fn set_residency(&mut self, block_idx: usize, residency: LayerResidency) {
        self.layers[block_idx] = residency;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_cpu_plan() {
        let plan = ResidencyPlan::all_cpu(32);
        assert_eq!(plan.num_layers(), 32);
        assert_eq!(plan.gpu_count(), 0);
        assert_eq!(plan.cpu_count(), 32);
        assert!(matches!(plan.residency(0), LayerResidency::Cpu));
    }

    #[test]
    fn static_split_plan() {
        let plan = ResidencyPlan::static_split(32, 8, 0);
        assert_eq!(plan.gpu_count(), 8);
        assert_eq!(plan.cpu_count(), 24);
        assert!(matches!(
            plan.residency(0),
            LayerResidency::Gpu { ordinal: 0 }
        ));
        assert!(matches!(
            plan.residency(7),
            LayerResidency::Gpu { ordinal: 0 }
        ));
        assert!(matches!(plan.residency(8), LayerResidency::Cpu));
    }

    #[test]
    fn all_gpu_plan() {
        let plan = ResidencyPlan::all_gpu(4, 1);
        assert_eq!(plan.gpu_count(), 4);
        assert_eq!(plan.cpu_count(), 0);
        assert!(matches!(
            plan.residency(3),
            LayerResidency::Gpu { ordinal: 1 }
        ));
    }

    #[test]
    fn device_spec_display() {
        assert_eq!(DeviceSpec::Cpu.to_string(), "cpu");
        assert_eq!(DeviceSpec::Cuda { ordinal: 0 }.to_string(), "cuda:0");
        assert_eq!(DeviceSpec::Cuda { ordinal: 3 }.to_string(), "cuda:3");
    }

    #[test]
    fn set_residency() {
        let mut plan = ResidencyPlan::all_cpu(4);
        plan.set_residency(2, LayerResidency::Gpu { ordinal: 0 });
        assert!(matches!(
            plan.residency(2),
            LayerResidency::Gpu { ordinal: 0 }
        ));
        assert!(matches!(plan.residency(1), LayerResidency::Cpu));
    }
}
