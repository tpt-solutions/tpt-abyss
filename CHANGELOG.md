# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI: `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test`.
- `cargo-deny` configuration for license/advisory/source checks, wired into CI.
- Root `README.md` with architecture diagram and quick-start.
- `CONTRIBUTING.md` with dev workflow and publishing order.

### Changed
- `tpt-abyss-memory` is now feature-gated in `tpt-abyss-cli` (default on;
  `--no-default-features` to disable) so core crates build without it.

## [0.1.0] - 2026-07-20

Initial workspace skeleton and first crates.

### Added
- `tpt-abyss-types`: `LayerProgram`, token/position types, `ReasoningTrace` /
  `VerificationResult`, shared error types.
- `tpt-abyss-router`: heuristic/rule-based dynamic-depth router with dependency-light
  math and a criterion latency benchmark harness.
- `tpt-abyss-engine`: `candle`-based GGUF loading and non-sequential forward pass
  executing arbitrary layer programs with correct per-layer KV-cache growth;
  router hook; per-layer activation logging.
- `tpt-abyss-verify`: in-process symbolic verifier (parser, arithmetic, logic,
  confidence scoring) with a plain `verify` API.
- `tpt-abyss-memory`: embedded `redb` storage for reasoning traces, a causal
  graph, an in-process vector index, and time-series quality tracking.
- `tpt-abyss-cli`: `generate` / `solve` (test-time compute loop) / `bench`
  subcommands demonstrating end-to-end dynamic-depth, verified inference.
