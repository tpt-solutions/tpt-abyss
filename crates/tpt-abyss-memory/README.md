# tpt-abyss-memory

Embedded **persistent memory** for
[TPT Abyss](https://github.com/tpt-solutions/tpt-abyss) — the "TPT Keystone"
layer. Built on [`redb`](https://crates.io/crates/redb), a transactional
embedded store with **no external server**.

Three logical stores back the architecture:

- `reasoning_traces`: id, embedding, trace text, success score, task type,
  timestamp.
- `causal_relationships`: cause, effect, confidence, discovery session.
- `quality_timeseries`: per-task-type reasoning-quality tracking over time.

A small **in-process vector index** provides cosine similarity search over
stored embeddings with no external vector DB. Feature-gated via the
`persistent` feature so the core crates work without it.

## Example

```rust
use tpt_abyss_memory::storage::MemoryStore;

let m = MemoryStore::open_temp().unwrap();
// store reasoning traces, causal relationships, quality samples, and query
// similar traces via `m.similar_traces(&query_embedding, 5)`.
```

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
