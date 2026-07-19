# tpt-abyss-verify

From-scratch, in-process **symbolic reasoner / verifier** for
[TPT Abyss](https://github.com/tpt-solutions/tpt-abyss).

Given a free-text chain-of-thought, it:

1. **Parses** it into a structured [`ReasoningTrace`] (statement / computation /
   deduction / conclusion / goal steps, with arithmetic operands extracted).
2. **Checks arithmetic** (`12 * 3 = 36`) for correctness.
3. **Detects contradictions** (a later step negating an earlier numeric claim).
4. Returns a [`VerificationResult`] with a status, a confidence score, and a
   correction hint for the engine's regeneration loop.

The core entry point is a plain Rust call with **no network hop**:

```rust
use tpt_abyss_verify::{parse_trace, verify};
use tpt_abyss_types::VerificationStatus;

let trace = parse_trace("t1", "math", "3 * 12 = 36\nanswer: 36").unwrap();
let result = verify(&trace).unwrap();
assert_eq!(result.status, VerificationStatus::Consistent);
```

This is the "TPT Eve" cognitive co-processor in the TPT Abyss architecture:
`draft → verify → accept or regenerate`.

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
