//! Integration tests for spite-loader using fake GGUF fixtures.

use spite_loader::{GgufModel, config::ModelHyperparams};
use spite_testkit::FakeGguf;

#[test]
fn opens_valid_file() {
    let tmp = FakeGguf::default().write_to_tempfile().unwrap();
    GgufModel::open(tmp.path()).expect("should open valid GGUF");
}

#[test]
fn arch_is_read_from_metadata() {
    let tmp = FakeGguf::default().write_to_tempfile().unwrap();
    let model = GgufModel::open(tmp.path()).unwrap();
    assert_eq!(model.arch(), "llama");
}

#[test]
fn get_u32_returns_metadata_value() {
    let fake = FakeGguf { d_model: 128, n_layers: 4, ..Default::default() };
    let tmp = fake.write_to_tempfile().unwrap();
    let model = GgufModel::open(tmp.path()).unwrap();
    assert_eq!(model.get_u32("llama.embedding_length"), Some(128));
    assert_eq!(model.get_u32("llama.block_count"),      Some(4));
}

#[test]
fn get_f32_returns_rope_theta() {
    let fake = FakeGguf { rope_theta: 500_000.0, ..Default::default() };
    let tmp = fake.write_to_tempfile().unwrap();
    let model = GgufModel::open(tmp.path()).unwrap();
    let v = model.get_f32("llama.rope.freq_base").unwrap();
    assert!((v - 500_000.0).abs() < 1.0, "expected 500_000, got {v}");
}

#[test]
fn get_str_returns_architecture() {
    let tmp = FakeGguf::default().write_to_tempfile().unwrap();
    let model = GgufModel::open(tmp.path()).unwrap();
    assert_eq!(model.get_str("general.architecture"), Some("llama"));
}

#[test]
fn unknown_key_returns_none() {
    let tmp = FakeGguf::default().write_to_tempfile().unwrap();
    let model = GgufModel::open(tmp.path()).unwrap();
    assert_eq!(model.get_u32("does.not.exist"), None);
}

#[test]
fn tensor_not_present_returns_null() {
    let tmp = FakeGguf::default().write_to_tempfile().unwrap();
    let model = GgufModel::open(tmp.path()).unwrap();
    let t = model.tensor("nonexistent.weight");
    assert!(t.is_null());
}

#[test]
fn tensor_present_and_correct_shape() {
    let fake = FakeGguf { d_model: 64, d_ffn: 256, ..Default::default() };
    let tmp = fake.write_to_tempfile().unwrap();
    let model = GgufModel::open(tmp.path()).unwrap();

    let q = model.tensor("blk.0.attn_q.weight");
    assert!(!q.is_null(), "blk.0.attn_q.weight should exist");
    assert_eq!(q.ne[0], 64);
    assert_eq!(q.ne[1], 64);

    let gate = model.tensor("blk.0.ffn_gate.weight");
    assert!(!gate.is_null());
    assert_eq!(gate.ne[0], 256); // d_ffn
    assert_eq!(gate.ne[1], 64);  // d_model
}

#[test]
fn hyperparams_parse_from_fake_model() {
    let fake = FakeGguf {
        d_model:    128,
        n_layers:   4,
        n_heads:    4,
        n_kv_heads: 2,
        d_ffn:      512,
        vocab_size: 64,
        max_seq_len: 1024,
        ..Default::default()
    };
    let tmp = fake.write_to_tempfile().unwrap();
    let model = GgufModel::open(tmp.path()).unwrap();

    // ModelHyperparams::from_meta is the bridge the executor uses
    let hp = ModelHyperparams::from_gguf(&model);
    assert_eq!(hp.arch,        "llama");
    assert_eq!(hp.d_model,     128);
    assert_eq!(hp.n_layers,    4);
    assert_eq!(hp.n_heads,     4);
    assert_eq!(hp.n_kv_heads,  2);
    assert_eq!(hp.d_ffn,       512);
    assert_eq!(hp.vocab_size,  64);
    assert_eq!(hp.max_seq_len, 1024);
}

#[test]
fn nonexistent_file_returns_error() {
    let result = GgufModel::open("/tmp/spite-does-not-exist-xyz.gguf");
    assert!(result.is_err());
}
