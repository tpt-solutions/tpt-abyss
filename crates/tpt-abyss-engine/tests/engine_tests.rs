use candle_core::Device;
use tpt_abyss_engine::synthetic::synthetic_model;
use tpt_abyss_engine::{forward_program, Engine, EngineConfig, KvCachePool};
use tpt_abyss_types::{LayerId, LayerProgram};

fn small_model(device: &Device) -> tpt_abyss_engine::ModelWeights {
    synthetic_model(4, 16, 4, 2, 4, 32, 64, device).unwrap()
}

#[test]
fn forward_sequential_matches_depth() {
    let dev = Device::Cpu;
    let model = small_model(&dev);
    let mut kv = KvCachePool::new(
        model.cfg.num_layers,
        model.cfg.num_kv_heads,
        model.cfg.head_dim,
        &dev,
    );

    let tokens = vec![1u32, 2, 3];
    let program = LayerProgram::sequential(4).unwrap();
    let (logits, acts) =
        forward_program(&model, &program, &tokens, 0, &mut kv, &dev, None).unwrap();
    assert_eq!(logits.dims(), &[64]);
    assert_eq!(acts.len(), 4, "one activation entry per executed layer");
}

#[test]
fn repeated_layer_grows_its_own_kv_cache() {
    let dev = Device::Cpu;
    let model = small_model(&dev);
    let mut kv = KvCachePool::new(
        model.cfg.num_layers,
        model.cfg.num_kv_heads,
        model.cfg.head_dim,
        &dev,
    );

    // Program repeats layer 3 (0-based index 2) twice: [1,2,3,3,4] (1-based).
    let program = LayerProgram::new(
        vec![LayerId(1), LayerId(2), LayerId(3), LayerId(3), LayerId(4)],
        4,
    )
    .unwrap();
    let tokens = vec![5u32, 6];
    let _ = forward_program(&model, &program, &tokens, 0, &mut kv, &dev, None).unwrap();

    // Layer 3 (0-based 2) ran twice => its KV cache length is seq(2)*2 = 4.
    assert_eq!(kv.layer(2).len(), 4, "repeated layer accumulates KV");
    // Other layers ran once => length 2.
    assert_eq!(kv.layer(0).len(), 2);
    assert_eq!(kv.layer(1).len(), 2);
    assert_eq!(kv.layer(3).len(), 2);
}

#[test]
fn non_sequential_program_changes_output_vs_sequential() {
    let dev = Device::Cpu;
    let model = small_model(&dev);

    let mut kv_seq = KvCachePool::new(
        model.cfg.num_layers,
        model.cfg.num_kv_heads,
        model.cfg.head_dim,
        &dev,
    );

    let tokens = vec![7u32, 8, 9];
    let (seq_logits, _) = forward_program(
        &model,
        &LayerProgram::sequential(4).unwrap(),
        &tokens,
        0,
        &mut kv_seq,
        &dev,
        None,
    )
    .unwrap();

    let mut kv_dyn = KvCachePool::new(
        model.cfg.num_layers,
        model.cfg.num_kv_heads,
        model.cfg.head_dim,
        &dev,
    );

    let program = LayerProgram::new(
        vec![LayerId(1), LayerId(2), LayerId(3), LayerId(3), LayerId(4)],
        4,
    )
    .unwrap();
    let (dyn_logits, _) =
        forward_program(&model, &program, &tokens, 0, &mut kv_dyn, &dev, None).unwrap();

    let a: Vec<f32> = seq_logits.to_vec1().unwrap();
    let b: Vec<f32> = dyn_logits.to_vec1().unwrap();
    let diff: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum();
    assert!(diff > 1e-4, "non-sequential program must alter logits");
}

#[test]
fn engine_generate_runs_and_logs_activations() {
    let dev = Device::Cpu;
    let model = small_model(&dev);
    let cfg = EngineConfig {
        max_context: 2048,
        temperature: 0.0,
        top_k: 40,
        top_p: 0.95,
        eos_token_id: 2,
    };
    let mut engine = Engine::from_weights(model, cfg);
    let out = engine.generate(&[1, 2, 3], 8, None).unwrap();
    assert_eq!(out.len(), 8);
    // Activation log should have one entry per generated step.
    assert_eq!(engine.activation_log().len(), 8);
}
