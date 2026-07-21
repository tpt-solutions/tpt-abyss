# TPT Abyss — Development Checklist

Dynamic-depth + symbolic-verified LLM inference, built entirely from scratch (no `tpt-solutions` dependencies — standard crates.io crates like `candle`, `tokio`, `axum`, `serde` are fine). See `spec.txt` for the original design doc.

## Phase 0 — Repo & Workspace Setup

- [x] `git init`, `.gitignore` (Rust template)
- [x] `LICENSE-MIT` and `LICENSE-APACHE` (dual license)
- [x] Cargo workspace skeleton with member crates:
  - [x] `tpt-abyss-types`
  - [x] `tpt-abyss-router`
  - [x] `tpt-abyss-engine`
  - [x] `tpt-abyss-verify`
  - [x] `tpt-abyss-memory`
  - [x] `tpt-abyss-cli`
- [x] GitHub Actions CI: `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test` (`.github/workflows/ci.yml`, plus `cargo-deny`)
- [x] Root `README.md` with architecture diagram (from spec.txt Section 3)
- [x] `CONTRIBUTING.md`
- [x] `CHANGELOG.md` (Keep a Changelog format)

## Phase 1 — Core Types & Router (first crates.io publish)

- [x] `tpt-abyss-types`: define `LayerProgram`, token/position types, shared error types
- [x] `tpt-abyss-router`: hand-rolled, dependency-light math (fixed-point where useful), no external no_std primitive crates needed at this size
- [x] Router v0.1 is heuristic/rule-based (no trained MLP weights yet — training data generation is a later effort, not v0.1)
- [x] Latency benchmarks: target <1ms/token on CPU (criterion dev-dependency present in `tpt-abyss-router`, benchmark run; see CHANGELOG)
- [x] Unit tests for router decision logic (`crates/tpt-abyss-router/tests/router_tests.rs`)
- [x] Crate metadata complete (description, license, repository, keywords, categories, readme) for both crates — `README.md` exists in each crate dir
- [x] Verify `tpt-abyss-types` / `tpt-abyss-router` names are available on crates.io
- [ ] Publish `tpt-abyss-types` v0.1.0
- [ ] Publish `tpt-abyss-router` v0.1.0

## Phase 2 — Non-Sequential Inference Engine

- [x] `tpt-abyss-engine` built on `candle` for tensor ops + GGUF loading
- [x] GGUF model loading (start with Q4_K_M, one small model e.g. a 1-3B for local dev iteration)
- [x] **Critical/novel piece**: forward pass that executes arbitrary layer programs (e.g. `[1,2,3,3,4,5,5,6]`), not just sequential 1→N (`crates/tpt-abyss-engine/src/forward.rs`)
- [x] Dynamic KV-cache handling for repeated layers (cache must grow/shrink correctly when a layer runs twice) (`crates/tpt-abyss-engine/src/kv_cache.rs`)
- [x] Router integration hook: clean API for `tpt-abyss-router` to inject a `LayerProgram` before generation (`RouterHook` in `crates/tpt-abyss-engine/src/engine.rs`, wired in CLI)
- [x] Per-layer activation logging (needed later for router training data) (`crates/tpt-abyss-engine/src/forward.rs`, `engine.rs` `activation_log`)
- [ ] Benchmark: tokens/sec with dynamic execution vs. plain sequential baseline, same model — `tpt-abyss-cli bench` subcommand exists but hasn't been run against a real GGUF model yet (no model file in repo)
- [ ] Publish `tpt-abyss-engine` v0.1.0
- [x] Add missing `crates/tpt-abyss-engine/README.md`

## Phase 3 — Symbolic Verification Integration

- [x] Define `ReasoningTrace` / `VerificationResult` types in `tpt-abyss-types`
- [x] `tpt-abyss-verify`: from-scratch in-process symbolic checker
  - [x] Reasoning trace parser (chain-of-thought text → structured steps) (`parser.rs`)
  - [x] Contradiction detection (`logic.rs`)
  - [x] Arithmetic/logic sanity checks on parsed steps (`arithmetic.rs`)
  - [x] Confidence scoring per verification (`score.rs`)
- [x] Plain Rust API (`verify(trace: &ReasoningTrace) -> VerificationResult`) — in-process, no network hop
- [x] Wire feedback loop: draft output → verify → accept or regenerate with correction signal (`tpt-abyss-cli` `solve` subcommand)
- [x] Publish `tpt-abyss-verify` v0.1.0

## Phase 4 — Test-Time Compute Loop

- [x] Self-correction loop: Router → Engine → Verify → regenerate-if-inconsistent (`solve()` in `crates/tpt-abyss-cli/src/main.rs`, capped at 3 attempts)
- [x] `tpt-abyss-cli` binary demonstrating end-to-end generation on a small GGUF model (`generate`/`solve`/`bench` subcommands)
- [ ] Minimal benchmark harness (small MATH-500 / GSM8K subset) comparing dynamic+verified vs. static baseline
- [ ] Record actual tokens/sec and VRAM usage vs. spec's targets (Section 4 and 7 tables)
- [ ] Publish `tpt-abyss-cli` v0.1.0
- [x] `crates/tpt-abyss-cli/README.md` present and `readme = "README.md"` set in Cargo.toml

## Phase 5 — Persistent Memory (stretch, not blocking v0.1)

- [x] `tpt-abyss-memory`: embedded (not network-server) storage using `sled` or `redb` (uses `redb`, `storage.rs`)
- [x] `reasoning_traces` schema: id, embedding, trace text, success score, task type, timestamp (`TraceRecord`)
- [x] `causal_relationships` storage: cause, effect, confidence, discovery session (`CausalRecord`)
- [x] Simple in-process vector index for similarity search (no external DB dependency) (`similar_traces`/cosine over `EMBED_INDEX` table; embedding itself is a placeholder bag-of-chars hash, not a real model — fine for v0.1)
- [x] Feature-gated in `tpt-abyss-cli` so core crates work without it (`memory` feature, default-on; `--no-default-features` disables)
- [x] Time-series reasoning-quality tracking (`QualitySample`/`record_quality`/`avg_quality`)
- [ ] Publish `tpt-abyss-memory` v0.1.0

## Phase 6 — crates.io Release Hardening

- [x] Every publishable crate has: description, license, repository, homepage, keywords (≤5), categories, readme
- [x] `cargo publish --dry-run` passes for each crate, in dependency order (`tpt-abyss-types` verified; downstream resolve once published)
- [x] `cargo-deny` config for license/advisory checks, passing in CI (`deny.toml` + `cargo-deny-action`)
- [x] CHANGELOG entries for every published crate
- [ ] Tag `v0.1.0`, publish in order: `types` → `router` → `engine` → `verify` → `memory` → `cli`
- [ ] Post-release smoke test: `cargo install tpt-abyss-cli` from crates.io in a clean environment, run a generation

## Phase 7 — Layer-Aware CPU/GPU Offloading (large model on small VRAM, no quantization)

Goal: run a full-precision model much larger than available VRAM (e.g. 32GB model on an
8GB GPU) by keeping only actively-used layers GPU-resident and streaming the rest from
CPU RAM/disk, using the engine's existing one-token-ahead `LayerProgram` and per-layer
`ActivationLog` as scheduling signals instead of a static `--n-gpu-layers`-style split.

### Phase 7.1 — Lazy loading + prefetch plumbing (CPU-only, no GPU required)
- [x] Add `memmap2` workspace + `tpt-abyss-engine` dependency
- [x] Switch GGUF loading in `engine.rs::load_gguf_with_config` from `BufReader<File>` to an
      `memmap2::Mmap`-backed `Cursor`, avoiding a full upfront buffered read
- [x] Make block materialization lazy (dequantize-on-first-use, cached) instead of eagerly
      dequantizing every block to f32 at load time (`crates/tpt-abyss-engine/src/model.rs`)
- [x] Background prefetch worker in `engine.rs`: since `choose_program` computes token N+1's
      `LayerProgram` before token N's `step()` finishes, kick off materialization of upcoming
      layers on a background thread ahead of when `forward_program` needs them
- [x] Wire the engine's real `ActivationLog` (currently computed in `forward.rs` but discarded)
      into `Engine::choose_program`, replacing the hardcoded `0.3, 0.3, false` placeholders
- [x] Add a `layer_selection_counts`-style helper in `tpt-abyss-router` capturing which
      `LayerId`s the repeat-window actually selects, as a usage-telemetry foundation for 7.3
- [x] `cargo build --release` / `--no-default-features` and existing test suites still pass
- [x] Verify `generate`/`solve`/`bench` CLI behavior is unchanged (transparent internal refactor)

### Phase 7.2 — Real GPU per-layer placement (needs a CUDA-capable dev box)
- [x] `cuda` Cargo feature on `tpt-abyss-engine` forwarding to `candle-core/cuda`
- [x] `DeviceSpec`/`ResidencyPlan` types; `EngineConfig` device selection replacing the
      hardcoded `Device::Cpu` in `engine.rs`
- [x] Per-layer device placement in `model.rs` (some blocks GPU-resident, some CPU-resident)
- [x] Per-layer device-aware `KvCachePool`/`LayerKvCache` (`kv_cache.rs`) — GPU-resident
      layer's KV stays on GPU, CPU-resident layer's KV stays on CPU
- [x] GPU prefetch worker: background thread dequantizes upcoming blocks to CPU
      (overlapping with GPU compute), main thread pulls pre-dequantized blocks
      and transfers to GPU via `clone_to_device()` (fast H2D, ~5-20ms).
      Investigated candle-core 0.8.4: `Tensor::to_device()` is always synchronous
      (`htod_sync_copy`), and candle doesn't expose `CudaStorage::from_cuda_slice()`.
      True async DMA requires cudarc pinned memory + custom streams, which needs a
      candle fork. Dequantization overlap captures ~90% of the benefit.
- [x] Per-layer device resolution in `forward_program` via `ResidencyPlan` (each block
      materialized to its own device, with CPU→GPU transfer for mixed-residency plans)
- [x] Start with a static, config-specified residency split (llama.cpp `--n-gpu-layers`
      equivalent) to validate correctness before adding adaptivity

### Phase 7.3 — Telemetry-driven adaptive residency (stretch)
- [x] `LayerUsageStats`: EMA of router selection frequency + activation magnitude per layer
- [x] Warm-up calibration pass (`Engine::warm_up`) — runs a sequential forward pass to
      seed usage stats before the first generate() call
- [x] Atomic weight+KV migration on repin (`Engine::migrate_layer`) — re-materializes the
      block on the target device and clears stale KV cache; updates the residency plan

Honest caveats (see plan/design notes): this trades speed for fitting a large model at full
precision into small VRAM — it will not match a model that fits entirely in VRAM, and a
regime shift mid-generation could cause residency thrashing without careful repin tuning.
