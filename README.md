# TPT Abyss

Dynamic-depth + symbolic-verified LLM inference, built from scratch on standard
crates.io dependencies (`candle`, `tokio`, `clap`, `serde`, …). No proprietary
`*tpt-*` runtime dependencies are used — the architecture described in
[`spec.txt`](./spec.txt) (TPT Spark/Eve/Keystone/…) is reimplemented here as
self-contained Rust crates.

The core idea: instead of running transformer layers `1 → N` in fixed order,
TPT Abyss executes an arbitrary **layer program** (e.g. `[1,2,3,3,4,5,5,6]`) so
that "hard" tokens can be given more compute via layer repetition, then a
symbolic verifier checks the reasoning and triggers regeneration when it is
inconsistent.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        tpt-abyss-cli                              │
│   generate · solve (test-time compute loop) · bench               │
└───────────────┬───────────────────┬──────────────────┬───────────┘
                │                   │                  │
        ┌───────▼───────┐   ┌───────▼───────┐  ┌───────▼──────────┐
        │ tpt-abyss-    │   │ tpt-abyss-    │  │ tpt-abyss-       │
        │ router        │   │ engine        │  │ verify (TPT Eve) │
        │ LayerProgram  │   │ non-sequential│  │ symbolic checker │
        │ per token     │   │ forward + KV  │  │ parse/arithmetic │
        └───────┬───────┘   │ cache pool    │  │ /contradiction   │
                │           └───────┬───────┘  └───────┬──────────┘
                │                   │                  │
                └─────────┬─────────┴──────────────────┘
                          │
                  ┌───────▼───────┐   ┌──────────────────────┐
                  │ tpt-abyss-    │   │ tpt-abyss-memory     │
                  │ types         │   │ (optional, redb)     │
                  │ shared types  │   │ trace/causal/quality │
                  └───────────────┘   └──────────────────────┘
```

- **`tpt-abyss-types`** — shared types: `LayerProgram`, token/position types,
  `ReasoningTrace` / `VerificationResult`, error types.
- **`tpt-abyss-router`** — dependency-light, CPU-friendly router that emits a
  per-token `LayerProgram`. v0.1 is heuristic/rule-based (no trained weights).
- **`tpt-abyss-engine`** — `candle`-based inference engine. Loads GGUF models
  and runs **arbitrary layer programs** with correct per-layer KV-cache growth.
- **`tpt-abyss-verify`** — in-process symbolic verifier ("TPT Eve"): parses a
  chain-of-thought, checks arithmetic, detects contradictions, returns a
  confidence score + correction hint.
- **`tpt-abyss-memory`** — embedded `redb` storage for reasoning traces, a
  causal graph, an in-process vector index, and time-series quality tracking.
  Feature-gated in the CLI (`--no-default-features` to disable).
- **`tpt-abyss-cli`** — end-to-end binary exposing `generate`, `solve`, and
  `bench`.

## Inference flow

```
Input → Router → LayerProgram → Engine (non-sequential forward) → Draft output
                                                    │
                                            tpt-abyss-verify
                                                    │
                                          Consistent? → Final output
                                              ↓ No
                                      Correction signal → Regenerate
```

## Quick start

```bash
# Build everything
cargo build --release

# Generate with a dynamic layer program
cargo run --bin tpt-abyss -- generate --model models/llama-3.2-3b-q4_k_m.gguf --prompt "Explain dynamic depth."

# Run the self-correcting test-time compute loop
cargo run --bin tpt-abyss -- solve --model models/llama-3.2-3b-q4_k_m.gguf --prompt "If 3x + 5 = 20, what is x?"

# Benchmark dynamic vs. sequential tokens/sec
cargo run --bin tpt-abyss -- bench --model models/llama-3.2-3b-q4_k_m.gguf
```

The memory subsystem is optional:

```bash
cargo build --release --no-default-features
```

## Project status

See [`TODO.md`](./TODO.md) for the phased task list. Phases 0–5 are largely
implemented; remaining work is benchmarking against real models, router
training-data collection, and crates.io publication.

## License

Dual-licensed under either of [MIT](./LICENSE-MIT) or
[Apache-2.0](./LICENSE-APACHE) at your option.
