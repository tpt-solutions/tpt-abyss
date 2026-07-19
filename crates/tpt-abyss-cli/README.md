# tpt-abyss-cli

End-to-end command-line interface for
[TPT Abyss](https://github.com/tpt-solutions/tpt-abyss): dynamic-depth
generation with symbolic verification and (optionally) persistent memory.

## Subcommands

- `generate` — generate text from a prompt using a dynamic-depth layer program,
  or a static sequential baseline (`--sequential`) for comparison.
- `solve` — run the self-correcting **test-time compute loop**: generate a
  reasoning trace, verify it symbolically, and regenerate with a correction
  signal if it is inconsistent (capped at 3 attempts).
- `bench` — benchmark tokens/sec for the dynamic program vs. a sequential
  baseline on the loaded model.

## Usage

```bash
# Build
cargo build --release

# Generate (dynamic depth)
cargo run --bin tpt-abyss -- generate \
  --model models/llama-3.2-3b-q4_k_m.gguf \
  --prompt "Explain dynamic depth in transformers."

# Self-correcting solve loop
cargo run --bin tpt-abyss -- solve \
  --model models/llama-3.2-3b-q4_k_m.gguf \
  --prompt "If 3x + 5 = 20, what is x?"

# Benchmark
cargo run --bin tpt-abyss -- bench --model models/llama-3.2-3b-q4_k_m.gguf
```

Model and tokenizer paths can also be set via the `TPT_MODEL` / `TPT_TOKENIZER`
environment variables.

## Features

- `memory` (default) — enables the `tpt-abyss-memory` persistent-memory crate.
  Build with `--no-default-features` to omit it; the core engine/router/verify
  paths still work.

## License

Dual-licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
