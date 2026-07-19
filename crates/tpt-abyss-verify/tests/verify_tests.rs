use tpt_abyss_types::VerificationStatus;
use tpt_abyss_verify::{parse_trace, verify};

#[test]
fn parses_and_verifies_correct_trace() {
    let text = "\
goal: find total cost
we have 3 items at 12 each
3 * 12 = 36
therefore total = 36
answer: 36";
    let trace = parse_trace("t1", "math", text).unwrap();
    assert_eq!(trace.steps.len(), 4);
    let result = verify(&trace).unwrap();
    assert_eq!(result.status, VerificationStatus::Consistent);
    assert!(result.violations.is_empty());
    assert!(result.confidence > 0.7);
}

#[test]
fn detects_arithmetic_error() {
    let text = "\
goal: multiply
4 * 5 = 25
answer: 25";
    let trace = parse_trace("t2", "math", text).unwrap();
    let result = verify(&trace).unwrap();
    assert_eq!(result.status, VerificationStatus::Inconsistent);
    assert!(result
        .violations
        .iter()
        .any(|v| v.kind == "arithmetic_error"));
}

#[test]
fn detects_contradiction() {
    let text = "\
let x = 10
now x = 20
answer: 20";
    let trace = parse_trace("t3", "logic", text).unwrap();
    let result = verify(&trace).unwrap();
    assert_eq!(result.status, VerificationStatus::Inconsistent);
    assert!(result.violations.iter().any(|v| v.kind == "contradiction"));
}

#[test]
fn division_by_zero_flagged() {
    let text = "5 / 0 = 0";
    let trace = parse_trace("t4", "math", text).unwrap();
    let result = verify(&trace).unwrap();
    assert!(result
        .violations
        .iter()
        .any(|v| v.kind == "division_by_zero"));
}
