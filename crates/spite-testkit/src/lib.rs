//! Test helpers for spite — fake GGUF files, fixture builders.
//!
//! Use as a dev-dependency:
//!
//! ```toml
//! [dev-dependencies]
//! spite-testkit = { workspace = true }
//! ```
//!
//! Then in tests:
//!
//! ```rust,ignore
//! let gguf = FakeGguf::default().write_to_tempfile();
//! let model = GgufModel::open(gguf.path()).unwrap();
//! assert_eq!(model.arch(), "llama4");
//! ```

use std::io;
use std::path::Path;

use tempfile::NamedTempFile;

// ── GGUF binary constants (must match spite-loader) ────────────────────────

const GGUF_MAGIC: u32 = 0x46554747; // b"GGUF" little-endian
const GGUF_VERSION: u32 = 3;

// value type tags
const VTYPE_U32: u32 = 4;
const VTYPE_F32: u32 = 6;
const VTYPE_STR: u32 = 8;
const VTYPE_ARRAY: u32 = 9;

// tensor type tag for F32
const TTYPE_F32: u32 = 0;

// data section alignment (must match loader)
const ALIGNMENT: usize = 32;

// ── Public API ─────────────────────────────────────────────────────────────

/// Minimal model configuration used to generate a fake GGUF file.
///
/// All dimensions are tiny by default so tests run in microseconds
/// on CPU without any GPU and without any real model weights.
///
/// The generated file is structurally identical to a real GGUF —
/// it passes every loader check — but weight tensors contain zeros.
#[derive(Debug, Clone)]
pub struct FakeGguf {
    pub arch: String,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub d_model: u32,
    pub d_ffn: u32,
    pub vocab_size: u32,
    pub max_seq_len: u32,
    pub rope_theta: f32,
    pub norm_eps: f32,
}

impl Default for FakeGguf {
    fn default() -> Self {
        Self {
            arch: "llama4".to_string(),
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            d_model: 64,
            d_ffn: 128,
            vocab_size: 32,
            max_seq_len: 512,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
        }
    }
}

impl FakeGguf {
    /// Write the fake GGUF to `path`.
    pub fn write(&self, path: &Path) -> io::Result<()> {
        let bytes = self.encode();
        std::fs::write(path, bytes)
    }

    /// Write to a named temp file; the file is deleted when dropped.
    pub fn write_to_tempfile(&self) -> io::Result<NamedTempFile> {
        let tmp = tempfile::Builder::new().suffix(".gguf").tempfile()?;
        self.write(tmp.path())?;
        Ok(tmp)
    }

    // ── Encoding ──────────────────────────────────────────────────────────

    pub fn encode(&self) -> Vec<u8> {
        let tensors = self.tensor_list();
        let kvs = self.kv_list();

        let mut buf = Vec::with_capacity(4096);

        // ── header ────────────────────────────────────────────────────────
        push_u32(&mut buf, GGUF_MAGIC);
        push_u32(&mut buf, GGUF_VERSION);
        push_u64(&mut buf, tensors.len() as u64);
        push_u64(&mut buf, kvs.len() as u64);

        // ── metadata KV pairs ─────────────────────────────────────────────
        for (key, val) in &kvs {
            push_str(&mut buf, key);
            match val {
                Kv::U32(v) => {
                    push_u32(&mut buf, VTYPE_U32);
                    push_u32(&mut buf, *v);
                }
                Kv::F32(v) => {
                    push_u32(&mut buf, VTYPE_F32);
                    push_f32(&mut buf, *v);
                }
                Kv::Str(s) => {
                    push_u32(&mut buf, VTYPE_STR);
                    push_str(&mut buf, s);
                }
                Kv::Arr(items) => {
                    push_u32(&mut buf, VTYPE_ARRAY);
                    // Homogeneous element type from the first item.
                    let elem = match items.first() {
                        Some(Kv::U32(_)) => VTYPE_U32,
                        Some(Kv::F32(_)) => VTYPE_F32,
                        _ => VTYPE_STR,
                    };
                    push_u32(&mut buf, elem);
                    push_u64(&mut buf, items.len() as u64);
                    for item in items {
                        match item {
                            Kv::U32(v) => push_u32(&mut buf, *v),
                            Kv::F32(v) => push_f32(&mut buf, *v),
                            Kv::Str(s) => push_str(&mut buf, s),
                            Kv::Arr(_) => {}
                        }
                    }
                }
            }
        }

        // ── tensor info ───────────────────────────────────────────────────
        // First pass: compute byte offsets per tensor
        let mut offsets = Vec::with_capacity(tensors.len());
        let mut running = 0u64;
        for t in &tensors {
            offsets.push(running);
            running += t.byte_size();
        }
        let total_data = running as usize;

        for (t, &off) in tensors.iter().zip(offsets.iter()) {
            push_str(&mut buf, &t.name);
            push_u32(&mut buf, t.dims.len() as u32); // ndim
            for &d in &t.dims {
                push_u64(&mut buf, d);
            }
            push_u32(&mut buf, TTYPE_F32);
            push_u64(&mut buf, off);
        }

        // ── padding to 32-byte alignment ──────────────────────────────────
        let pad = (ALIGNMENT - (buf.len() % ALIGNMENT)) % ALIGNMENT;
        buf.extend(std::iter::repeat_n(0u8, pad));

        // ── tensor data (all zeros) ────────────────────────────────────────
        buf.extend(std::iter::repeat_n(0u8, total_data));

        buf
    }

    fn kv_list(&self) -> Vec<(String, Kv)> {
        let a = &self.arch;
        vec![
            ("general.architecture".into(), Kv::Str(a.clone())),
            (format!("{a}.block_count"), Kv::U32(self.n_layers)),
            (format!("{a}.embedding_length"), Kv::U32(self.d_model)),
            (format!("{a}.attention.head_count"), Kv::U32(self.n_heads)),
            (
                format!("{a}.attention.head_count_kv"),
                Kv::U32(self.n_kv_heads),
            ),
            (format!("{a}.feed_forward_length"), Kv::U32(self.d_ffn)),
            (format!("{a}.context_length"), Kv::U32(self.max_seq_len)),
            (format!("{a}.rope.freq_base"), Kv::F32(self.rope_theta)),
            (
                format!("{a}.attention.layer_norm_rms_epsilon"),
                Kv::F32(self.norm_eps),
            ),
            (
                "tokenizer.ggml.token_count".into(),
                Kv::U32(self.vocab_size),
            ),
            ("tokenizer.ggml.model".into(), Kv::Str("gpt2".into())),
            (
                "tokenizer.ggml.tokens".into(),
                Kv::Arr(fake_tokens(self.vocab_size)),
            ),
            (
                "tokenizer.ggml.scores".into(),
                Kv::Arr(fake_scores(self.vocab_size)),
            ),
            (
                "tokenizer.ggml.token_type".into(),
                Kv::Arr(fake_types(self.vocab_size)),
            ),
            (
                "tokenizer.ggml.merges".into(),
                Kv::Arr(vec![Kv::Str("A B".into())]),
            ),
            ("tokenizer.ggml.bos_token_id".into(), Kv::U32(0)),
            ("tokenizer.ggml.eos_token_id".into(), Kv::U32(1)),
        ]
    }

    fn tensor_list(&self) -> Vec<Tensor> {
        let d = self.d_model as u64;
        let ff = self.d_ffn as u64;
        let v = self.vocab_size as u64;

        let mut t = vec![
            Tensor::new("token_embd.weight", vec![d, v]),
            Tensor::new("output_norm.weight", vec![d]),
            Tensor::new("output.weight", vec![d, v]),
        ];

        for i in 0..self.n_layers {
            let b = format!("blk.{i}");
            t.push(Tensor::new(format!("{b}.attn_norm.weight"), vec![d]));
            t.push(Tensor::new(format!("{b}.ffn_norm.weight"), vec![d]));
            t.push(Tensor::new(format!("{b}.attn_q.weight"), vec![d, d]));
            t.push(Tensor::new(format!("{b}.attn_k.weight"), vec![d, d]));
            t.push(Tensor::new(format!("{b}.attn_v.weight"), vec![d, d]));
            t.push(Tensor::new(format!("{b}.attn_output.weight"), vec![d, d]));
            t.push(Tensor::new(format!("{b}.ffn_gate.weight"), vec![d, ff]));
            t.push(Tensor::new(format!("{b}.ffn_up.weight"), vec![d, ff]));
            t.push(Tensor::new(format!("{b}.ffn_down.weight"), vec![ff, d]));
        }

        t
    }
}

// ── Internal helpers ───────────────────────────────────────────────────────

enum Kv {
    U32(u32),
    F32(f32),
    Str(String),
    Arr(Vec<Kv>),
}

/// Tiny test vocabulary: ids 0/1 are BOS/EOS sentinels, then the exact
/// bytes needed to encode chat-style prompts ("user: AB"), then A-Z
/// filler, with the last id doubling as the "AB" merge result.
fn fake_tokens(n: u32) -> Vec<Kv> {
    let mut toks = vec![
        Kv::Str("<bos>".into()),
        Kv::Str("<eos>".into()),
        Kv::Str("u".into()),
        Kv::Str("s".into()),
        Kv::Str("e".into()),
        Kv::Str("r".into()),
        Kv::Str(":".into()),
        Kv::Str(" ".into()),
        Kv::Str("A".into()),
        Kv::Str("B".into()),
    ];
    for i in toks.len() as u32..n {
        let b = b'A' + ((i as u8) % 26);
        toks.push(Kv::Str((b as char).to_string()));
    }
    // Last token doubles as the "AB" merge result.
    if let Some(last) = toks.last_mut() {
        *last = Kv::Str("AB".into());
    }
    toks.truncate(n as usize);
    toks
}

fn fake_scores(n: u32) -> Vec<Kv> {
    (0..n).map(|_| Kv::F32(0.0)).collect()
}

fn fake_types(n: u32) -> Vec<Kv> {
    (0..n).map(|_| Kv::U32(1)).collect()
}

struct Tensor {
    name: String,
    dims: Vec<u64>,
}

impl Tensor {
    fn new(name: impl Into<String>, dims: Vec<u64>) -> Self {
        Self {
            name: name.into(),
            dims,
        }
    }

    fn byte_size(&self) -> u64 {
        self.dims.iter().product::<u64>() * 4 // F32
    }
}

fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_str(buf: &mut Vec<u8>, s: &str) {
    push_u64(buf, s.len() as u64);
    buf.extend_from_slice(s.as_bytes());
}

// ── Tests for the testkit itself ───────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use spite_loader::GgufModel;

    #[test]
    fn roundtrip_default() {
        let tmp = FakeGguf::default().write_to_tempfile().unwrap();
        let model = GgufModel::open(tmp.path()).expect("loader should accept fake GGUF");
        assert_eq!(model.arch(), "llama4");
        assert_eq!(model.get_u32("llama4.block_count"), Some(2));
        assert_eq!(model.get_u32("llama4.embedding_length"), Some(64));
        assert_eq!(model.get_u32("llama4.attention.head_count"), Some(2));
        assert_eq!(model.get_u32("tokenizer.ggml.token_count"), Some(32));
    }

    #[test]
    fn roundtrip_custom_arch() {
        let fake = FakeGguf {
            arch: "mistral".to_string(),
            n_layers: 4,
            ..Default::default()
        };
        let tmp = fake.write_to_tempfile().unwrap();
        let model = GgufModel::open(tmp.path()).unwrap();
        assert_eq!(model.arch(), "mistral");
        assert_eq!(model.get_u32("mistral.block_count"), Some(4));
    }

    #[test]
    fn tensor_names_present() {
        let tmp = FakeGguf::default().write_to_tempfile().unwrap();
        let model = GgufModel::open(tmp.path()).unwrap();
        // Layer 0 tensors should be readable
        let t = model.tensor("blk.0.attn_q.weight");
        assert!(!t.is_null());
        assert_eq!(t.ne[0], 64);
        assert_eq!(t.ne[1], 64);
    }

    #[test]
    fn tensor_count_matches() {
        let fake = FakeGguf {
            n_layers: 2,
            ..Default::default()
        };
        let tmp = fake.write_to_tempfile().unwrap();
        let model = GgufModel::open(tmp.path()).unwrap();
        // 3 global + 9 per layer × 2 layers = 21
        assert_eq!(model.n_tensors(), 21);
    }

    #[test]
    fn bad_magic_rejected() {
        let mut bytes = FakeGguf::default().encode();
        bytes[0] = 0xFF; // corrupt magic
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), bytes).unwrap();
        assert!(GgufModel::open(tmp.path()).is_err());
    }
}
