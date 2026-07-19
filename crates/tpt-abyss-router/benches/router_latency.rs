use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_abyss_router::{HeuristicRouter, RouterConfig};
use tpt_abyss_types::{Position, TokenId};

fn bench_route_token(c: &mut Criterion) {
    let router = HeuristicRouter::new(RouterConfig::default());
    c.bench_function("route_token_easy", |b| {
        b.iter(|| {
            let p = router.route_token(
                black_box(TokenId(42)),
                black_box(Position(5)),
                black_box(0.1),
                black_box(0.1),
                black_box(false),
            );
            black_box(p)
        })
    });
    c.bench_function("route_token_hard", |b| {
        b.iter(|| {
            let p = router.route_token(
                black_box(TokenId(9999)),
                black_box(Position(500)),
                black_box(0.95),
                black_box(0.9),
                black_box(true),
            );
            black_box(p)
        })
    });
}

criterion_group!(benches, bench_route_token);
criterion_main!(benches);
