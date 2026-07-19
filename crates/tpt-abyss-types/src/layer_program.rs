use crate::{AbyssError, AbyssResult};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::Deref;

/// A 1-based model layer index (model layer 1 ..= N).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LayerId(pub u32);

impl LayerId {
    /// Construct a layer id from a 0-based raw index.
    pub fn from_zero_based(raw: u32) -> Self {
        LayerId(raw + 1)
    }

    /// The 0-based index used for weight tensor selection.
    pub fn as_zero_based(self) -> u32 {
        self.0.saturating_sub(1)
    }
}

impl fmt::Display for LayerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L{}", self.0)
    }
}

/// An execution plan: an ordered list of layer indices to run.
///
/// A `LayerProgram` is what makes TPT Abyss non-sequential. Instead of
/// running every layer 1..=N once, the router emits an arbitrary program
/// such as `[1, 2, 3, 3, 4, 5, 5, 6]` where layer 3 and 5 are repeated to
/// allocate more compute to "hard" tokens. Layer 1 is conventionally the
/// embedding/input projection; layer N the final norm/head projection.
///
/// Programs are 1-based and validated to be non-empty and within
/// `[1, model_depth]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerProgram {
    /// 1-based layer ids, in execution order.
    steps: Vec<LayerId>,
    /// The depth (max layer index) the program is valid against.
    model_depth: u32,
}

impl LayerProgram {
    /// Build a validated program against the given model depth.
    ///
    /// # Errors
    /// Returns [`AbyssError::InvalidLayerProgram`] if `steps` is empty or any
    /// step is out of the `[1, model_depth]` range.
    pub fn new(steps: Vec<LayerId>, model_depth: u32) -> AbyssResult<Self> {
        if steps.is_empty() {
            return Err(AbyssError::InvalidLayerProgram {
                reason: "program must contain at least one layer".to_string(),
            });
        }
        for &s in &steps {
            if s.0 == 0 || s.0 > model_depth {
                return Err(AbyssError::InvalidLayerProgram {
                    reason: format!("layer {s} out of range 1..={model_depth}"),
                });
            }
        }
        Ok(Self { steps, model_depth })
    }

    /// The total number of layer executions (program length).
    #[inline]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the program is empty (always false; see [`LayerProgram::new`]).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// The model depth this program is valid against.
    #[inline]
    pub fn model_depth(&self) -> u32 {
        self.model_depth
    }

    /// Iterate over the layer ids in execution order.
    #[inline]
    pub fn steps(&self) -> impl Iterator<Item = LayerId> + '_ {
        self.steps.iter().copied()
    }

    /// The raw step slice.
    #[inline]
    pub fn as_slice(&self) -> &[LayerId] {
        &self.steps
    }

    /// The maximum number of times any single layer is repeated. A value of 1
    /// means the program is fully sequential.
    pub fn max_repeat_count(&self) -> usize {
        self.steps
            .iter()
            .copied()
            .fold(std::collections::HashMap::new(), |mut acc, l| {
                *acc.entry(l).or_insert(0usize) += 1;
                acc
            })
            .into_values()
            .max()
            .unwrap_or(0)
    }

    /// True if every layer 1..=model_depth appears exactly once, in order.
    pub fn is_sequential(&self) -> bool {
        self.steps.len() as u32 == self.model_depth
            && self.steps.windows(2).all(|w| w[1].0 == w[0].0 + 1)
    }

    /// Build the standard sequential program `1..=model_depth`.
    pub fn sequential(model_depth: u32) -> AbyssResult<Self> {
        let steps = (1..=model_depth).map(LayerId).collect();
        Self::new(steps, model_depth)
    }
}

impl Deref for LayerProgram {
    type Target = [LayerId];
    fn deref(&self) -> &Self::Target {
        &self.steps
    }
}

impl fmt::Display for LayerProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self.steps.iter().map(|l| l.to_string()).collect();
        write!(f, "[{}]", parts.join(", "))
    }
}

/// Ergonomic builder for [`LayerProgram`] that also accepts raw integers.
pub struct LayerProgramBuilder {
    steps: Vec<LayerId>,
    model_depth: u32,
}

impl LayerProgramBuilder {
    /// Start a builder against the given model depth.
    pub fn with_depth(model_depth: u32) -> Self {
        Self {
            steps: Vec::new(),
            model_depth,
        }
    }

    /// Append a layer by 1-based id.
    pub fn layer(mut self, id: u32) -> Self {
        self.steps.push(LayerId(id));
        self
    }

    /// Append a raw 0-based layer index.
    pub fn raw(mut self, id: u32) -> Self {
        self.steps.push(LayerId::from_zero_based(id));
        self
    }

    /// Append a repeated run of `count` copies of `id`.
    pub fn repeat(mut self, id: u32, count: usize) -> Self {
        self.steps
            .extend(std::iter::repeat(LayerId(id)).take(count));
        self
    }

    /// Finalize, validating against `model_depth`.
    pub fn build(self) -> AbyssResult<LayerProgram> {
        LayerProgram::new(self.steps, self.model_depth)
    }
}
