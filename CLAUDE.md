# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TPT Abyss: dynamic-depth, symbolic-verified LLM inference in Rust, built from
scratch on standard crates.io deps (`candle`, `tokio`, `clap`, `redb`, …) — no
proprietary `tpt-*` runtime dependencies. The core idea: instead of running
transformer layers `1 → N` in fixed sequence, the engine executes an arbitrary
**layer program** (e.g. `[1,2,3,3,4,5,5,6]`) so "hard" tokens get more compute
via layer repetition, then an in-process symbolic verifier checks the
reasoning and can trigger regeneration. See `spec.txt` for the original design
doc (`TODO.md` tracks phased implementation status against it).

## Commands

```bash
# Build (memory subsystem is default-on; do NOT pass --all-features here —
# it pulls in tpt-abyss-engine's `cuda` feature, which needs nvcc/CUDA
# toolkit and fails on regular dev boxes and CI runners)
cargo build --workspace
cargo build --release --no-default-features   # without tpt-abyss-memory

# Format / lint — CI fails on either
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings

# Tests
cargo test --workspace
cargo test -p tpt-abyss-engine                 # single crate
cargo test -p tpt-abyss-router router_tests     # single test file/name filter

# Router latency benchmark (the only crate with criterion benches currently)
cargo bench -p tpt-abyss-router

# License/advisory check (mirrors CI's cargo-deny job)
cargo deny check

# On a CUDA-capable dev box only, opt into GPU layer placement:
cargo build -p tpt-abyss-engine --features cuda

# Run the CLI (needs a local GGUF model, e.g. under models/)
cargo run --bin tpt-abyss -- generate --model models/<file>.gguf --prompt "..."
cargo run --bin tpt-abyss -- solve --model models/<file>.gguf --prompt "..."
cargo run --bin tpt-abyss -- bench --model models/<file>.gguf
```

CI (`.github/workflows/ci.yml`) runs exactly the fmt/clippy/build/test commands
above plus `cargo-deny`, against `.github/deny.toml` — reproduce locally
before pushing.

## Crate architecture

Six-crate workspace, each independently publishable to crates.io, in strict
dependency order `types → router → engine → verify → memory → cli`:

- **`tpt-abyss-types`** — shared types with no logic: `LayerProgram`,
  token/position types, `ReasoningTrace` / `VerificationResult`, error types
  (`AbyssError`/`AbyssResult`). Every other crate depends on this one.
- **`tpt-abyss-router`** (`heuristic.rs`, `math.rs`, `features.rs`) —
  dependency-light, CPU-friendly. `HeuristicRouter::route_token` /
  `route_features` turn per-token features (entropy, residual magnitude, …)
  into a `LayerProgram`. v0.1 is rule-based, no trained MLP weights yet —
  training-data collection is future work (Phase 7.3-ish, see `TODO.md`).
- **`tpt-abyss-engine`** (`engine.rs`, `forward.rs`, `kv_cache.rs`,
  `model.rs`, `device_placement.rs`, `usage_stats.rs`) — `candle`-based. Loads
  GGUF weights and is the one genuinely novel piece: `forward.rs` executes an
  arbitrary layer program rather than a fixed 1→N pass, and `kv_cache.rs`
  implements a per-layer KV-cache pool that must grow/shrink correctly when a
  layer index repeats within a program. `Engine::step`/`generate` in
  `engine.rs` wires in the router via `choose_program` (computes token N+1's
  program while token N is still finishing, for prefetch purposes) and
  records `ActivationLog` per layer for router-training signal.
  `device_placement.rs`/`usage_stats.rs` are scaffolding for Phase 7 CPU/GPU
  layer offloading (see `TODO.md` Phase 7) — not yet wired into the hot path.
- **`tpt-abyss-verify`** ("TPT Eve": `parser.rs`, `logic.rs`,
  `arithmetic.rs`, `score.rs`) — in-process symbolic checker, no network hop.
  `parser.rs` turns chain-of-thought text into structured steps, `logic.rs`
  detects contradictions (`extract_claim` parses `var = value` style claims —
  mind qualifier words like "let"/"now" before the variable), `arithmetic.rs`
  does sanity checks, `score.rs` produces a confidence score. Entry point:
  `verify(trace: &ReasoningTrace) -> AbyssResult<VerificationResult>` in
  `lib.rs`.
- **`tpt-abyss-memory`** (`storage.rs`) — embedded `redb` storage (not a
  network server): `reasoning_traces`, `causal_relationships`, an in-process
  cosine-similarity vector index (embeddings are currently a placeholder
  bag-of-chars hash, not a real embedding model), and time-series quality
  tracking. Feature-gated behind the `memory` feature in the CLI, default-on;
  `open_temp()` backs onto a real unique temp file (redb needs a concrete
  path) and removes it on drop.
- **`tpt-abyss-cli`** (`main.rs`, `bench_harness.rs`) — binary `tpt-abyss`
  with `generate` / `solve` / `bench` / `evaluate` subcommands (`Commands` enum
  in `main.rs`). `solve` implements the test-time compute loop: Router →
  Engine → Verify → regenerate-on-inconsistency, capped at 3 attempts.

## Inference flow

```
Input → Router (LayerProgram) → Engine (non-sequential forward, per-layer KV)
                                              │
                                       draft output
                                              │
                                     tpt-abyss-verify
                                              │
                          consistent? ──yes──→ final output
                                │no
                    correction signal → regenerate (solve loop, ≤3 attempts)
```

## Conventions

- Crate metadata: every publishable crate needs `description`, `license`,
  `repository`, `homepage`, `keywords` (≤5), `categories`, `readme`. Run
  `cargo publish --dry-run` before considering a crate release-ready.
  Workspace-level fields (`version`, `edition`, `license`, `repository`,
  shared dep versions) live in the root `Cargo.toml` `[workspace.package]` /
  `[workspace.dependencies]` — prefer `{ workspace = true }` in member crates
  over pinning duplicate versions.
- Update `TODO.md` (phased checklist tracking implementation against
  `spec.txt`) and `CHANGELOG.md` (Keep a Changelog format) when completing a
  tracked task or changing user-facing behavior.
- No local GGUF model ships in the repo; `models/` is where you place one
  (e.g. a small 1–3B Q4_K_M) for local dev/testing — several benchmark and
  integration items in `TODO.md` are blocked on this.
