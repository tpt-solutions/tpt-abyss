//! Telemetry helpers for router selection tracking.
//!
//! `LayerSelectionStats` accumulates which `LayerId`s the router selects across
//! multiple program decisions, providing per-layer selection frequency counts.
//! This is the foundation for Phase 7.3's adaptive GPU residency: knowing which
//! layers are "hot" (frequently selected) vs "cold" (rarely selected) lets the
//! engine pin hot layers to GPU-resident memory.

use std::collections::HashMap;
use tpt_abyss_types::LayerId;

/// Per-layer selection frequency counter.
///
/// Tracks how many times each `LayerId` appears across a series of router
/// program decisions. Provides helpers for extracting the most/least selected
/// layers, which is the basis for adaptive GPU residency in Phase 7.3.
#[derive(Debug, Clone, Default)]
pub struct LayerSelectionStats {
    /// Selection count per LayerId (1-based).
    counts: HashMap<LayerId, usize>,
    /// Total number of programs recorded.
    total_programs: usize,
    /// Total number of layer executions across all programs.
    total_executions: usize,
}

impl LayerSelectionStats {
    /// Create an empty stats accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a single program's layer selections. Increments the count for
    /// each `LayerId` that appears in the program.
    pub fn record(&mut self, program: &tpt_abyss_types::LayerProgram) {
        self.total_programs += 1;
        for &layer in program.as_slice() {
            *self.counts.entry(layer).or_insert(0) += 1;
            self.total_executions += 1;
        }
    }

    /// Selection count for a specific layer.
    pub fn count(&self, layer: LayerId) -> usize {
        self.counts.get(&layer).copied().unwrap_or(0)
    }

    /// Selection frequency for a layer (count / total_executions).
    /// Returns 0.0 if no programs have been recorded.
    pub fn frequency(&self, layer: LayerId) -> f32 {
        if self.total_executions == 0 {
            return 0.0;
        }
        self.count(layer) as f32 / self.total_executions as f32
    }

    /// All layers with their selection counts, sorted by count descending.
    pub fn ranked(&self) -> Vec<(LayerId, usize)> {
        let mut entries: Vec<_> = self.counts.iter().map(|(&l, &c)| (l, c)).collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries
    }

    /// The K most frequently selected layers (hot layers).
    pub fn hot_layers(&self, k: usize) -> Vec<LayerId> {
        self.ranked().into_iter().take(k).map(|(l, _)| l).collect()
    }

    /// The K least frequently selected layers (cold layers).
    pub fn cold_layers(&self, k: usize) -> Vec<LayerId> {
        let mut ranked = self.ranked();
        ranked.reverse();
        ranked.into_iter().take(k).map(|(l, _)| l).collect()
    }

    /// Total number of programs recorded.
    pub fn total_programs(&self) -> usize {
        self.total_programs
    }

    /// Total layer executions across all recorded programs.
    pub fn total_executions(&self) -> usize {
        self.total_executions
    }

    /// Reset all counters.
    pub fn clear(&mut self) {
        self.counts.clear();
        self.total_programs = 0;
        self.total_executions = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_abyss_types::LayerProgram;

    #[test]
    fn records_layer_frequencies() {
        let mut stats = LayerSelectionStats::new();
        // Program: [1,2,3,3,4] — layer 3 selected twice.
        let prog = LayerProgram::new(
            vec![LayerId(1), LayerId(2), LayerId(3), LayerId(3), LayerId(4)],
            4,
        )
        .unwrap();
        stats.record(&prog);

        assert_eq!(stats.count(LayerId(3)), 2);
        assert_eq!(stats.count(LayerId(1)), 1);
        assert_eq!(stats.total_executions(), 5);
        assert_eq!(stats.total_programs(), 1);
    }

    #[test]
    fn hot_and_cold_layers() {
        let mut stats = LayerSelectionStats::new();
        let prog1 = LayerProgram::sequential(4).unwrap();
        stats.record(&prog1);
        // Second record: repeat layer 4 twice.
        let prog2 = LayerProgram::new(
            vec![LayerId(1), LayerId(2), LayerId(3), LayerId(4), LayerId(4)],
            4,
        )
        .unwrap();
        stats.record(&prog2);

        let hot = stats.hot_layers(1);
        assert_eq!(hot, vec![LayerId(4)]); // 3 total (1+2)
                                           // cold_layers returns least-selected; with counts {1:2, 2:2, 3:2, 4:3}
                                           // any of layers 1-3 could be returned first (all have count 2).
        let cold = stats.cold_layers(1);
        assert_eq!(cold.len(), 1);
        assert!(
            cold[0] == LayerId(1) || cold[0] == LayerId(2) || cold[0] == LayerId(3),
            "cold layer should be one with count 2, got {:?}",
            cold[0]
        );
    }

    #[test]
    fn frequency_calculation() {
        let mut stats = LayerSelectionStats::new();
        let prog = LayerProgram::new(vec![LayerId(1), LayerId(1), LayerId(2)], 2).unwrap();
        stats.record(&prog);
        // Layer 1: 2/3, Layer 2: 1/3
        let f1 = stats.frequency(LayerId(1));
        let f2 = stats.frequency(LayerId(2));
        assert!((f1 - 2.0 / 3.0).abs() < 1e-6);
        assert!((f2 - 1.0 / 3.0).abs() < 1e-6);
    }
}
