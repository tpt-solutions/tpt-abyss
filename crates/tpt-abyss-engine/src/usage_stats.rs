//! Telemetry-driven layer usage statistics (Phase 7.3).
//!
//! `LayerUsageStats` combines two signals per layer into a single "usage score"
//! via Exponential Moving Average (EMA):
//!
//! 1. **Selection frequency** — how often the router selects this layer across
//!    program decisions (from `LayerSelectionStats`).
//! 2. **Activation magnitude** — mean absolute activation from the forward pass
//!    (from `ActivationLog`).
//!
//! The combined score is used for adaptive GPU residency: hot layers (high
//! usage) are pinned to GPU, cold layers are evicted to CPU. The EMA decay
//! ensures recent behavior dominates while old data fades smoothly.

use crate::forward::ActivationLog;
use std::collections::HashMap;
use tpt_abyss_types::LayerId;

/// Per-layer usage statistics computed via exponential moving average.
///
/// After each generation step, call `update(activation_log, selection_counts)`
/// to incorporate the latest signal. The `usage_score` for each layer then
/// reflects a smoothed blend of recent selection frequency and activation
/// magnitude.
#[derive(Debug, Clone)]
pub struct LayerUsageStats {
    /// EMA of selection frequency per layer (0.0 .. 1.0).
    selection_ema: HashMap<LayerId, f32>,
    /// EMA of activation magnitude per layer (0.0 .. 1.0, normalized).
    activation_ema: HashMap<LayerId, f32>,
    /// Combined usage score per layer (blend of selection and activation).
    usage_scores: HashMap<LayerId, f32>,
    /// EMA decay factor. Higher = more weight on recent observations.
    /// Typical: 0.1 .. 0.3.
    decay: f32,
    /// Weight given to selection frequency vs activation magnitude in the
    /// combined score. 0.0 = all activation, 1.0 = all selection.
    selection_weight: f32,
    /// Number of update rounds processed.
    rounds: usize,
}

impl Default for LayerUsageStats {
    fn default() -> Self {
        Self::new(0.2, 0.5)
    }
}

impl LayerUsageStats {
    /// Create a new stats accumulator.
    ///
    /// `decay` controls EMA speed: 0.1 = slow (heavy smoothing), 0.5 = fast.
    /// `selection_weight` controls the blend: 0.5 gives equal weight to
    /// selection frequency and activation magnitude.
    pub fn new(decay: f32, selection_weight: f32) -> Self {
        Self {
            selection_ema: HashMap::new(),
            activation_ema: HashMap::new(),
            usage_scores: HashMap::new(),
            decay: decay.clamp(0.01, 1.0),
            selection_weight: selection_weight.clamp(0.0, 1.0),
            rounds: 0,
        }
    }

    /// Update statistics with one generation step's data.
    ///
    /// - `activation_log`: per-layer activation magnitudes from the forward pass.
    /// - `program_length`: the total number of layer executions in the program.
    ///   Used to normalize selection counts to a 0..1 frequency.
    pub fn update(&mut self, activation_log: &ActivationLog, program_length: usize) {
        self.rounds += 1;
        let alpha = self.decay;

        // Compute per-layer selection frequency for this program.
        let mut sel_counts: HashMap<LayerId, usize> = HashMap::new();
        for &(layer, _) in activation_log {
            *sel_counts.entry(layer).or_insert(0) += 1;
        }
        let max_count = program_length.max(1) as f32;

        // Collect all known layers (from prior rounds + current).
        let all_layers: Vec<LayerId> = {
            let mut set: std::collections::HashSet<LayerId> =
                self.selection_ema.keys().copied().collect();
            set.extend(sel_counts.keys());
            set.into_iter().collect()
        };

        // Update selection EMA for ALL known layers. Layers not selected this
        // round get freq=0, which decays their EMA toward zero.
        for &layer in &all_layers {
            let freq = sel_counts.get(&layer).copied().unwrap_or(0) as f32 / max_count;
            let prev = self.selection_ema.get(&layer).copied().unwrap_or(0.0);
            let ema = alpha * freq + (1.0 - alpha) * prev;
            self.selection_ema.insert(layer, ema);
        }

        // Collect activation magnitudes for current round.
        let mut act_mags: HashMap<LayerId, f32> = HashMap::new();
        for &(layer, mag) in activation_log {
            act_mags.entry(layer).or_insert(0.0_f32);
            // Keep the maximum magnitude for this layer this round.
            let entry = act_mags.get_mut(&layer).unwrap();
            *entry = (*entry).max(mag.abs());
        }

        // Update activation EMA for ALL known layers.
        for &layer in &all_layers {
            let norm_mag = act_mags.get(&layer).copied().unwrap_or(0.0).min(1.0);
            let prev = self.activation_ema.get(&layer).copied().unwrap_or(0.0);
            let ema = alpha * norm_mag + (1.0 - alpha) * prev;
            self.activation_ema.insert(layer, ema);
        }

        // Recompute combined usage scores.
        self.recompute_scores();
    }

    /// Recompute all usage scores from the current EMA values.
    fn recompute_scores(&mut self) {
        let w = self.selection_weight;
        let all_layers: std::collections::HashSet<LayerId> = self
            .selection_ema
            .keys()
            .chain(self.activation_ema.keys())
            .copied()
            .collect();
        for layer in all_layers {
            let sel = self.selection_ema.get(&layer).copied().unwrap_or(0.0);
            let act = self.activation_ema.get(&layer).copied().unwrap_or(0.0);
            self.usage_scores.insert(layer, w * sel + (1.0 - w) * act);
        }
    }

    /// The combined usage score for a specific layer (0.0 .. 1.0).
    pub fn usage_score(&self, layer: LayerId) -> f32 {
        self.usage_scores.get(&layer).copied().unwrap_or(0.0)
    }

    /// The selection frequency EMA for a specific layer.
    pub fn selection_ema(&self, layer: LayerId) -> f32 {
        self.selection_ema.get(&layer).copied().unwrap_or(0.0)
    }

    /// The activation magnitude EMA for a specific layer.
    pub fn activation_ema(&self, layer: LayerId) -> f32 {
        self.activation_ema.get(&layer).copied().unwrap_or(0.0)
    }

    /// All layers sorted by usage score descending (hot layers first).
    pub fn ranked(&self) -> Vec<(LayerId, f32)> {
        let mut entries: Vec<_> = self.usage_scores.iter().map(|(&l, &s)| (l, s)).collect();
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entries
    }

    /// The K hottest layers (highest usage score).
    pub fn hot_layers(&self, k: usize) -> Vec<LayerId> {
        self.ranked().into_iter().take(k).map(|(l, _)| l).collect()
    }

    /// The K coldest layers (lowest usage score).
    pub fn cold_layers(&self, k: usize) -> Vec<LayerId> {
        let mut ranked = self.ranked();
        ranked.reverse();
        ranked.into_iter().take(k).map(|(l, _)| l).collect()
    }

    /// Number of update rounds processed.
    pub fn rounds(&self) -> usize {
        self.rounds
    }

    /// Reset all statistics.
    pub fn clear(&mut self) {
        self.selection_ema.clear();
        self.activation_ema.clear();
        self.usage_scores.clear();
        self.rounds = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_abyss_types::LayerId;

    #[test]
    fn tracks_usage_after_updates() {
        let mut stats = LayerUsageStats::new(0.5, 0.5);
        // Simulate: layer 1 appears 3x, layer 2 appears 1x.
        let log: ActivationLog = vec![
            (LayerId(1), 0.1),
            (LayerId(1), 0.2),
            (LayerId(1), 0.3),
            (LayerId(2), 0.05),
        ];
        stats.update(&log, 4);

        assert!(stats.usage_score(LayerId(1)) > stats.usage_score(LayerId(2)));
        assert_eq!(stats.rounds(), 1);
    }

    #[test]
    fn ema_smoothing() {
        let mut stats = LayerUsageStats::new(0.3, 1.0); // all selection weight
                                                        // Round 1: layer 1 selected 100%
        let log1: ActivationLog = vec![(LayerId(1), 0.1), (LayerId(1), 0.1)];
        stats.update(&log1, 2);
        let score_after_1 = stats.usage_score(LayerId(1));

        // Round 2: layer 2 selected 100% — layer 1 should decay but not vanish.
        let log2: ActivationLog = vec![(LayerId(2), 0.1), (LayerId(2), 0.1)];
        stats.update(&log2, 2);
        let score_layer1_after_2 = stats.usage_score(LayerId(1));
        // Layer 1 still has some score due to EMA smoothing.
        assert!(
            score_layer1_after_2 > 0.0,
            "EMA should preserve prior signal"
        );
        assert!(score_layer1_after_2 < score_after_1, "Layer 1 should decay");
    }

    #[test]
    fn hot_and_cold() {
        let mut stats = LayerUsageStats::new(0.5, 0.5);
        // Layer 3 is always selected with high activation.
        for _ in 0..5 {
            let log: ActivationLog = vec![(LayerId(3), 0.9), (LayerId(1), 0.1)];
            stats.update(&log, 2);
        }
        let hot = stats.hot_layers(1);
        assert_eq!(hot, vec![LayerId(3)]);
    }
}
