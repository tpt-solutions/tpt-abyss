use tpt_abyss_memory::storage::*;
use tpt_abyss_types::AbyssError;

fn rec(id: &str, text: &str, score: f32, task: &str) -> TraceRecord {
    TraceRecord {
        id: id.to_string(),
        embedding: trivial_embedding(text, 16),
        trace_text: text.to_string(),
        success_score: score,
        task_type: task.to_string(),
        timestamp_ms: 0,
    }
}

#[test]
fn store_and_retrieve_trace() -> Result<(), AbyssError> {
    let m = MemoryStore::open_temp()?;
    let r = rec("a", "3 * 12 = 36", 1.0, "math");
    m.put_trace(&r)?;
    let got = m.get_trace("a")?;
    assert!(got.is_some());
    assert_eq!(got.unwrap().trace_text, "3 * 12 = 36");
    Ok(())
}

#[test]
fn similarity_search_ranks_near_text_higher() -> Result<(), AbyssError> {
    let m = MemoryStore::open_temp()?;
    m.put_trace(&rec("a", "multiply three by twelve", 1.0, "math"))?;
    m.put_trace(&rec("b", "the weather is sunny today", 0.0, "chitchat"))?;
    let query = trivial_embedding("three multiplied by twelve", 16);
    let hits = m.similar_traces(&query, 2)?;
    assert_eq!(hits[0].0, "a");
    Ok(())
}

#[test]
fn causal_and_quality_tracking() -> Result<(), AbyssError> {
    let m = MemoryStore::open_temp()?;
    m.put_causal(&CausalRecord {
        id: "c1".into(),
        cause: "x is even".into(),
        effect: "x+1 is odd".into(),
        confidence: 0.95,
        discovery_session: "s1".into(),
        timestamp_ms: 0,
    })?;
    assert_eq!(m.list_causal()?.len(), 1);

    m.record_quality("math", 0.8)?;
    m.record_quality("math", 0.9)?;
    assert!((m.avg_quality("math")?.unwrap() - 0.85).abs() < 1e-6);
    Ok(())
}

#[test]
fn cosine_basics() {
    assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
    assert!((cosine(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-6);
}
