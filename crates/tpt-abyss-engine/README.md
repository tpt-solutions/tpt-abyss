# tpt-abyss-engine

The **non-sequential (dynamic-depth) inference engine** for
[TPT Abyss](https://github.com/tpt-solutions/tpt-abyss), built on
[`candle`](https://crates.io/crates/candle-core).

## What makes it different

Most inference engines run transformer layers `1 → N` in fixed order. TPT Abyss
instead runs an arbitrary **layer program** — an ordered list of layer indices
such as `[1, 2, 3, 3, 4, 5, 5, 6]`. A layer that appears twice is executed
twice. This is the core "dynamic depth" mechanism: hard tokens can be given
more compute by repeating layers, without changing the model weights.

Because repeated layers accumulate their own key/value state, the engine keeps a
**per-layer KV cache** (`KvCachePool`) that grows and shrinks correctly as the
program is followed.

## Example

```rust
use tpt_abyss_engine::{Engine, EngineConfig};
use tpt_abyss_types::{LayerProgram, LayerId};

let mut engine = Engine::load_gguf("models/llama-3.2-3b-q4_k_m.gguf")?;

// Run a custom program: e.g. repeat layers 3 and 5.
let program = LayerProgram::try_from(vec![
    LayerId::new(1), LayerId::new(2), LayerId::new(3),
    LayerId::new(3), LayerId::new(4), LayerId::new(5),
    LayerId::new(5), LayerId::new(6),
])?;

let tokens = vec![1u32, 2, 3];
let (logits, acts) = engine.step(&tokens, 0, &program)?;
println!("generated {} logits; activation steps: {}", logits.len(), acts.len());
```

## Components

- `Engine` — GGUF loading, generation loop, router hook, activation logging.
- `forward_program` — the core non-sequential forward pass.
- `KvCachePool` / `LayerKvCache` — per-layer KV cache management.
- `ModelWeights` — lower-level model weights + GGUF loading.
- `ActivationLog` — per-layer activation magnitudes (for router-training data).

## License

Dual-licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
