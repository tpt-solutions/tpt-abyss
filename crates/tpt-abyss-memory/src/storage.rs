//! Embedded persistent memory for TPT Abyss.
//!
//! Built on [`redb`] (an embedded, transactional key-value / table store with
//! no external server). Three logical stores back the architecture's
//! persistent layer:
//!
//! - `reasoning_traces`: id, embedding, trace text, success score, task type,
//!   timestamp.
//! - `causal_relationships`: cause, effect, confidence, discovery session.
//! - `quality_timeseries`: per-task-type reasoning-quality tracking over time.
//!
//! A small in-process vector index over stored embeddings provides similarity
//! search without any external vector DB. Feature-gated via the `persistent`
//! feature so the core crates remain usable without the embedded DB.

use redb::{Database, ReadableTable, TableDefinition};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use tpt_abyss_types::AbyssError;

const TRACES: TableDefinition<&str, &[u8]> = TableDefinition::new("reasoning_traces");
const CAUSAL: TableDefinition<&str, &[u8]> = TableDefinition::new("causal_relationships");
const QUALITY: TableDefinition<&str, &[u8]> = TableDefinition::new("quality_timeseries");
const EMBED_INDEX: TableDefinition<&str, &[u8]> = TableDefinition::new("embedding_index");

/// A stored reasoning trace record.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TraceRecord {
    pub id: String,
    /// Fixed-dim embedding used for similarity retrieval (cosine).
    pub embedding: Vec<f32>,
    pub trace_text: String,
    /// 0.0 (failure) .. 1.0 (success).
    pub success_score: f32,
    pub task_type: String,
    pub timestamp_ms: u64,
}

/// A discovered causal relationship.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CausalRecord {
    pub id: String,
    pub cause: String,
    pub effect: String,
    pub confidence: f32,
    pub discovery_session: String,
    pub timestamp_ms: u64,
}

/// A single time-series quality sample.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QualitySample {
    pub task_type: String,
    pub score: f32,
    pub timestamp_ms: u64,
}

/// Handle to the persistent memory store.
pub struct MemoryStore {
    db: Database,
}

impl MemoryStore {
    /// Open (or create) a memory store at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, AbyssError> {
        let db = Database::create(path).map_err(|e| AbyssError::Memory(e.to_string()))?;
        // create tables eagerly
        let wtx = db
            .begin_write()
            .map_err(|e| AbyssError::Memory(e.to_string()))?;
        {
            wtx.open_table(TRACES)
                .map_err(|e| AbyssError::Memory(e.to_string()))?;
            wtx.open_table(CAUSAL)
                .map_err(|e| AbyssError::Memory(e.to_string()))?;
            wtx.open_table(QUALITY)
                .map_err(|e| AbyssError::Memory(e.to_string()))?;
            wtx.open_table(EMBED_INDEX)
                .map_err(|e| AbyssError::Memory(e.to_string()))?;
        }
        wtx.commit()
            .map_err(|e| AbyssError::Memory(e.to_string()))?;
        Ok(Self { db })
    }

    /// Open an in-memory (temporary) store, useful for tests.
    pub fn open_temp() -> Result<Self, AbyssError> {
        let db = Database::create("").map_err(|e| AbyssError::Memory(e.to_string()))?;
        let wtx = db
            .begin_write()
            .map_err(|e| AbyssError::Memory(e.to_string()))?;
        {
            wtx.open_table(TRACES)
                .map_err(|e| AbyssError::Memory(e.to_string()))?;
            wtx.open_table(CAUSAL)
                .map_err(|e| AbyssError::Memory(e.to_string()))?;
            wtx.open_table(QUALITY)
                .map_err(|e| AbyssError::Memory(e.to_string()))?;
            wtx.open_table(EMBED_INDEX)
                .map_err(|e| AbyssError::Memory(e.to_string()))?;
        }
        wtx.commit()
            .map_err(|e| AbyssError::Memory(e.to_string()))?;
        Ok(Self { db })
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Store a reasoning trace record.
    pub fn put_trace(&self, rec: &TraceRecord) -> Result<(), AbyssError> {
        let bytes = serde_json::to_vec(rec).map_err(AbyssError::Json)?;
        let wtx = self
            .db
            .begin_write()
            .map_err(|e| AbyssError::Memory(e.to_string()))?;
        let mut table = wtx
            .open_table(TRACES)
            .map_err(|e| AbyssError::Memory(e.to_string()))?;
        table
            .insert(rec.id.as_str(), bytes.as_slice())
            .map_err(|e| AbyssError::Memory(e.to_string()))?;
        // Mirror embedding into the index for similarity search.
        let mut emb = wtx
            .open_table(EMBED_INDEX)
            .map_err(|e| AbyssError::Memory(e.to_string()))?;
        let emb_bytes = serde_json::to_vec(&rec.embedding).map_err(AbyssError::Json)?;
        emb.insert(rec.id.as_str(), emb_bytes.as_slice())
            .map_err(|e| AbyssError::Memory(e.to_string()))?;
        drop(emb);
        drop(table);
        wtx.commit().map_err(|e| AbyssError::Memory(e.to_string()))
    }

    /// Retrieve a stored trace by id.
    pub fn get_trace(&self, id: &str) -> Result<Option<TraceRecord>, AbyssError> {
        let rtx = self
            .db
            .begin_read()
            .map_err(|e| AbyssError::Memory(e.to_string()))?;
        let table = rtx
            .open_table(TRACES)
            .map_err(|e| AbyssError::Memory(e.to_string()))?;
        match table
            .get(id)
            .map_err(|e| AbyssError::Memory(e.to_string()))?
        {
            Some(bytes) => {
                let rec = serde_json::from_slice(&bytes.value()).map_err(AbyssError::Json)?;
                Ok(Some(rec))
            }
            None => Ok(None),
        }
    }

    /// Simple in-process similarity search: returns stored trace ids ordered by
    /// cosine similarity to `query`, descending. `top_k` bounds the result.
    pub fn similar_traces(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<(String, f32)>, AbyssError> {
        let rtx = self
            .db
            .begin_read()
            .map_err(|e| AbyssError::Memory(e.to_string()))?;
        let table = rtx
            .open_table(EMBED_INDEX)
            .map_err(|e| AbyssError::Memory(e.to_string()))?;
        let mut scored: Vec<(String, f32)> = Vec::new();
        for entry in table
            .iter()
            .map_err(|e| AbyssError::Memory(e.to_string()))?
        {
            let (k, v) = entry.map_err(|e| AbyssError::Memory(e.to_string()))?;
            let emb: Vec<f32> = serde_json::from_slice(&v.value()).map_err(AbyssError::Json)?;
            let sim = cosine(query, &emb);
            scored.push((k.value().to_string(), sim));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }

    /// Store a causal relationship.
    pub fn put_causal(&self, rec: &CausalRecord) -> Result<(), AbyssError> {
        let bytes = serde_json::to_vec(rec).map_err(AbyssError::Json)?;
        let wtx = self
            .db
            .begin_write()
            .map_err(|e| AbyssError::Memory(e.to_string()))?;
        let mut table = wtx
            .open_table(CAUSAL)
            .map_err(|e| AbyssError::Memory(e.to_string()))?;
        table
            .insert(rec.id.as_str(), bytes.as_slice())
            .map_err(|e| AbyssError::Memory(e.to_string()))?;
        drop(table);
        wtx.commit().map_err(|e| AbyssError::Memory(e.to_string()))
    }

    /// All stored causal relationships.
    pub fn list_causal(&self) -> Result<Vec<CausalRecord>, AbyssError> {
        let rtx = self
            .db
            .begin_read()
            .map_err(|e| AbyssError::Memory(e.to_string()))?;
        let table = rtx
            .open_table(CAUSAL)
            .map_err(|e| AbyssError::Memory(e.to_string()))?;
        let mut out = Vec::new();
        for entry in table
            .iter()
            .map_err(|e| AbyssError::Memory(e.to_string()))?
        {
            let (_, v) = entry.map_err(|e| AbyssError::Memory(e.to_string()))?;
            out.push(serde_json::from_slice(&v.value()).map_err(AbyssError::Json)?);
        }
        Ok(out)
    }

    /// Record a reasoning-quality sample for time-series tracking.
    pub fn record_quality(&self, task_type: &str, score: f32) -> Result<(), AbyssError> {
        let sample = QualitySample {
            task_type: task_type.to_string(),
            score,
            timestamp_ms: Self::now_ms(),
        };
        let bytes = serde_json::to_vec(&sample).map_err(AbyssError::Json)?;
        let key = format!("{}:{}", task_type, sample.timestamp_ms);
        let wtx = self
            .db
            .begin_write()
            .map_err(|e| AbyssError::Memory(e.to_string()))?;
        let mut table = wtx
            .open_table(QUALITY)
            .map_err(|e| AbyssError::Memory(e.to_string()))?;
        table
            .insert(key.as_str(), bytes.as_slice())
            .map_err(|e| AbyssError::Memory(e.to_string()))?;
        drop(table);
        wtx.commit().map_err(|e| AbyssError::Memory(e.to_string()))
    }

    /// Average quality score for a task type (over all recorded samples).
    pub fn avg_quality(&self, task_type: &str) -> Result<Option<f32>, AbyssError> {
        let rtx = self
            .db
            .begin_read()
            .map_err(|e| AbyssError::Memory(e.to_string()))?;
        let table = rtx
            .open_table(QUALITY)
            .map_err(|e| AbyssError::Memory(e.to_string()))?;
        let mut sum = 0.0f32;
        let mut n = 0u32;
        for entry in table
            .iter()
            .map_err(|e| AbyssError::Memory(e.to_string()))?
        {
            let (_, v) = entry.map_err(|e| AbyssError::Memory(e.to_string()))?;
            let s: QualitySample = serde_json::from_slice(&v.value()).map_err(AbyssError::Json)?;
            if s.task_type == task_type {
                sum += s.score;
                n += 1;
            }
        }
        Ok(if n > 0 { Some(sum / n as f32) } else { None })
    }
}

/// Cosine similarity between two equal-or-unequal length vectors (zero if
/// either norm is zero).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

/// Produce a trivial deterministic embedding from text (bag-of-chars hashed
/// into a fixed dim). A stand-in until a real embedding model is wired in.
pub fn trivial_embedding(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    for c in text.chars() {
        v[(c as usize) % dim] += 1.0;
    }
    // L2 normalize.
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}
