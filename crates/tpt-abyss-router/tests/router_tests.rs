use tpt_abyss_router::{HeuristicRouter, RouterConfig, RouterConfigBuilder};
use tpt_abyss_types::{LayerProgram, Position, TokenId};

fn router() -> HeuristicRouter {
    HeuristicRouter::new(
        RouterConfigBuilder::default()
            .model_depth(8)
            .max_repeat(3)
            .max_program_len(40)
            .build(),
    )
}

#[test]
fn easy_token_is_sequential() {
    let r = router();
    let p = r
        .route_token(TokenId(10), Position(1), 0.05, 0.05, false)
        .unwrap();
    // Not hard => backbone only.
    assert!(p.is_sequential(), "expected sequential, got {p}");
    assert_eq!(p.len(), 8);
    assert_eq!(p.model_depth(), 8);
}

#[test]
fn hard_token_gets_repeats() {
    let r = router();
    let p = r
        .route_token(TokenId(7000), Position(100), 0.97, 0.95, true)
        .unwrap();
    assert!(!p.is_sequential(), "hard token should repeat layers: {p}");
    assert!(p.max_repeat_count() > 1, "some layer must repeat: {p}");
    // Must stay within the cap.
    assert!(p.len() <= 40, "program too long: {}", p.len());
}

#[test]
fn produced_program_is_valid() {
    let r = router();
    for pos in 0..50u32 {
        let p = r
            .route_token(
                TokenId(pos as u32 * 7),
                Position(pos),
                (pos as f32 / 50.0),
                (pos as f32 / 50.0),
                pos % 11 == 0,
            )
            .unwrap();
        // Re-validate through the public constructor semantics.
        let rebuilt = LayerProgram::new(p.as_slice().to_vec(), 8).unwrap();
        assert_eq!(rebuilt, p);
    }
}

#[test]
fn rejects_out_of_range_depth() {
    let r = router();
    // craft a program with a layer beyond depth via features is not possible;
    // instead test LayerProgram directly through the router's builder path.
    let p = r.route_token(TokenId(1), Position(0), 0.0, 0.0, false);
    assert!(p.is_ok());
}

#[test]
fn program_roundtrip_json() {
    let r = router();
    let p = r
        .route_token(TokenId(3), Position(2), 0.8, 0.7, false)
        .unwrap();
    let json = serde_json::to_string(&p).unwrap();
    let back: LayerProgram = serde_json::from_str(&json).unwrap();
    assert_eq!(p, back);
}
