# tpt-abyss-router

Dynamic depth router for [TPT Abyss](https://github.com/tpt-solutions/tpt-abyss).

The router converts a per-token feature vector into a [`LayerProgram`] — an
ordered list of layer indices to execute for that token (e.g.
`[1, 2, 3, 3, 4, 5, 5, 6]`). This is what enables **non-sequential
inference**: hard tokens get repeated layers (more compute), easy tokens run a
shallow backbone.

For `v0.1` the router is **heuristic / rule-based** (no trained MLP weights
yet). The policy is a pure function of interpretable per-token features, so it
can later be swapped for a small learned MLP with identical I/O. Design goals
per `tpt-abyss-router.txt`:

- **tiny** — no heavy external primitive crates at this size
- **fast** — `route_token` runs in well under 1 ms on CPU (see `benches/`)
- **panic-free** — all indexing is bounds-checked

## Example

```rust
use tpt_abyss_router::{HeuristicRouter, RouterConfig};
use tpt_abyss_types::{Position, TokenId};

let router = HeuristicRouter::new(RouterConfig::default());
let program = router
    .route_token(TokenId(7000), Position(100), 0.97, 0.9, true)
    .unwrap();
println!("layer program: {program}");
```

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
