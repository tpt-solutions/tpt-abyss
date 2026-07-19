# Contributing to TPT Abyss

Thanks for your interest in contributing! This document describes how to get set
up and the conventions we follow.

## Getting started

1. Fork and clone the repo.
2. Install a recent stable Rust toolchain (≥ `1.80`):
   ```bash
   rustup toolchain install stable
   ```
3. Build the workspace:
   ```bash
   cargo build --workspace --all-features
   ```

## Development workflow

- **Formatting**: run `cargo fmt --all` before committing. CI fails on
  `cargo fmt --check`.
- **Lints**: run `cargo clippy --all-targets --all-features -- -D warnings`.
  CI treats warnings as errors.
- **Tests**: run `cargo test --workspace --all-features`.
- **Benchmarks** (router only, currently):
  ```bash
  cargo bench -p tpt-abyss-router
  ```

## Commit / PR conventions

- Keep PRs focused; describe the *why*, not just the *what*.
- Add or update tests for behavioral changes.
- Update [`TODO.md`](./TODO.md) and [`CHANGELOG.md`](./CHANGELOG.md) when you
  complete a tracked task or change user-facing behavior.
- Dual-license: by contributing you agree your contributions are licensed under
  MIT OR Apache-2.0 (see [`LICENSE-MIT`](./LICENSE-MIT) and
  [`LICENSE-APACHE`](./LICENSE-APACHE)).

## Crate layout

Each crate under `crates/` is published independently. When adding a new
publishable crate:

- Include `description`, `license`, `repository`, `homepage`, `keywords` (≤5),
  `categories`, and a `README.md`.
- Run `cargo publish --dry-run` and confirm it passes.
- Publish in dependency order:
  `types → router → engine → verify → memory → cli`.

## Running the CLI

See the root [`README.md`](./README.md) for `generate` / `solve` / `bench`
usage. A small GGUF model (1–3B, Q4_K_M) is recommended for local iteration.
