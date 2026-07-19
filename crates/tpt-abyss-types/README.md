# tpt-abyss-types

Shared types for the TPT Abyss dynamic-depth inference stack.

This crate is the foundation that the other `tpt-abyss-*` crates depend on. It
defines crate-agnostic types with no heavy dependencies:

- `LayerProgram` — the ordered list of layer indices to execute
  (e.g. `[1, 2, 3, 3, 4, 5, 5, 6]`), the core of non-sequential inference.
- Token / position primitives.
- `ReasoningTrace` / `VerificationResult` — the neural-symbolic feedback-loop
  payloads.
- `AbyssError` — the crate-wide error type.

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
