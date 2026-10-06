//! Qwen3.8 Multi-Token Prediction (NextN) draft head — CPU reference.
//!
//! One extra dense Qwen3 block stored after the trunk as `blk.{n_layers}`:
//!
//! ```text
//! x  = eh_proj · [rmsnorm(embed(tok), enorm) || rmsnorm(h_trunk, hnorm)]
//! x += attn(rmsnorm(x, attn_norm))     QK-norm, NEOX RoPE, its own KV cache
//! x += ffn(rmsnorm(x, ffn_norm))       SwiGLU
//! logits = lm_head(rmsnorm(x, shared_head_norm | output_norm))
//! ```
//!
//! The GPU path runs the same block through the dispatch table: `mtp_stem`
//! for the two norms and the pack, `matmul` for `eh_proj` and the LM head, and
//! the trunk's `attention` / `ffn` ops for the block itself.

use std::sync::RwLock;

use spite_kvcache::{KvQuant, KvQuantConfig, VbrPolicy, VbrRows};
use spite_loader::GgufModel;

use super::qwen3_5::{LayerState, full_attn, qwen3_head_dim};
use crate::dense::{DenseWeights, matvec, rmsnorm};
use crate::{ModelConfig, ModelError};

pub struct Qwen3_8Mtp {
    config: ModelConfig,
    weights: Option<DenseWeights>,
    kv_quant: RwLock<KvQuantConfig>,
    state: RwLock<Option<LayerState>>,
}

impl Qwen3_8Mtp {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            weights: None,
            kv_quant: RwLock::new(KvQuantConfig::full_precision()),
            state: RwLock::new(None),
        }
    }

    pub fn load_weights(&mut self, model: &GgufModel) -> Result<(), ModelError> {
        self.weights = Some(DenseWeights::load(model)?);
        Ok(())
    }

    /// Drop the draft head's KV history (new sequence, or after a rejection
    /// rolled the trunk back past the draft positions).
    pub fn reset_cache(&self) {
        if let Ok(mut s) = self.state.write() {
            *s = None;
        }
    }

    /// Store the draft head's KV rows at `cfg`; clears the history.
    pub fn set_kv_quant(&self, cfg: KvQuantConfig) {
        if let Ok(mut q) = self.kv_quant.write() {
            *q = cfg;
        }
        self.reset_cache();
    }

    /// One draft step.
    ///
    /// - `prev_token`: the token at `pos` (sampled by the trunk, or by the
    ///   previous draft step when chaining)
    /// - `trunk_hidden`: hidden state that produced `prev_token` (`d_model`),
    ///   before the final norm
    /// - `pos`: sequence position of `prev_token`; calls must be sequential
    ///   because each one appends a row to the head's KV cache
    /// - `logits_out`: `vocab_size` draft logits for position `pos + 1`
    /// - `next_hidden_out`: optional `d_model` block output (pre-norm), the
    ///   `trunk_hidden` for the next chained draft step
    pub fn forward_step(
        &self,
        prev_token: u32,
        trunk_hidden: &[f32],
        pos: usize,
        logits_out: &mut [f32],
        next_hidden_out: Option<&mut [f32]>,
    ) -> Result<(), ModelError> {
        let Some(w) = &self.weights else {
            return Err(ModelError::Forward("load_weights not called".into()));
        };
        let cfg = &self.config;
        let d = cfg.d_model;
        let n_heads = cfg.n_heads.max(1);
        let n_kv_heads = cfg.n_kv_heads.max(1);
        let head_dim = qwen3_head_dim(cfg);
        let eps = cfg.norm_eps;
        if trunk_hidden.len() != d {
            return Err(ModelError::Forward(format!(
                "trunk_hidden length {} != d_model {d}",
                trunk_hidden.len()
            )));
        }
        if logits_out.len() != cfg.vocab_size {
            return Err(ModelError::Forward(format!(
                "logits_out length {} != vocab_size {}",
                logits_out.len(),
                cfg.vocab_size
            )));
        }
        if next_hidden_out.as_ref().is_some_and(|h| h.len() != d) {
            return Err(ModelError::Forward(
                "next_hidden_out length != d_model".into(),
            ));
        }

        let b = format!("blk.{}", cfg.n_layers);
        let embd = w.get("token_embd.weight")?;
        let tok = prev_token as usize;
        if tok >= embd.rows() || embd.cols() != d {
            return Err(ModelError::Forward(format!(
                "token {tok} outside token_embd ({} rows)",
                embd.rows()
            )));
        }

        // Stem: the two NextN norms, packed [e || h], projected back to d.
        let mut stem = vec![0f32; 2 * d];
        let (e_half, h_half) = stem.split_at_mut(d);
        rmsnorm(
            &embd.data[tok * d..(tok + 1) * d],
            &w.get(&format!("{b}.nextn.enorm.weight"))?.data,
            eps,
            e_half,
        );
        rmsnorm(
            trunk_hidden,
            &w.get(&format!("{b}.nextn.hnorm.weight"))?.data,
            eps,
            h_half,
        );
        let mut x = vec![0f32; d];
        matvec(w.get(&format!("{b}.nextn.eh_proj.weight"))?, &stem, &mut x)?;

        // Attention sub-layer.
        let mut n = vec![0f32; d];
        rmsnorm(
            &x,
            &w.get(&format!("{b}.attn_norm.weight"))?.data,
            eps,
            &mut n,
        );
        let (kq, vq) = self
            .kv_quant
            .read()
            .map(|c| (c.key, c.val))
            .unwrap_or((KvQuant::F32, KvQuant::F32));
        let mut state = self
            .state
            .write()
            .map_err(|_| ModelError::Forward("state lock".into()))?;
        let layer = state.get_or_insert_with(|| LayerState {
            k: VbrRows::new(
                n_kv_heads * head_dim,
                VbrPolicy::from_ctx(cfg.max_seq_len, kq),
            ),
            v: VbrRows::new(
                n_kv_heads * head_dim,
                VbrPolicy::from_ctx(cfg.max_seq_len, vq),
            ),
        });
        let attn = full_attn(w, &b, &n, pos, layer, cfg, head_dim, n_heads, n_kv_heads)?;
        drop(state);
        for (xi, a) in x.iter_mut().zip(attn) {
            *xi += a;
        }

        // SwiGLU sub-layer.
        rmsnorm(
            &x,
            &w.get(&format!("{b}.ffn_norm.weight"))?.data,
            eps,
            &mut n,
        );
        let w_gate = w.get(&format!("{b}.ffn_gate.weight"))?;
        let mut gate = vec![0f32; w_gate.rows()];
        let mut up = vec![0f32; w_gate.rows()];
        matvec(w_gate, &n, &mut gate)?;
        matvec(w.get(&format!("{b}.ffn_up.weight"))?, &n, &mut up)?;
        for (g, &u) in gate.iter_mut().zip(up.iter()) {
            *g = *g / (1.0 + (-*g).exp()) * u;
        }
        let mut down = vec![0f32; d];
        matvec(w.get(&format!("{b}.ffn_down.weight"))?, &gate, &mut down)?;
        for (xi, dl) in x.iter_mut().zip(down) {
            *xi += dl;
        }

        // Shared head (falls back to the trunk's final norm / tied embeddings).
        let head_norm = w
            .get(&format!("{b}.nextn.shared_head_norm.weight"))
            .or_else(|_| w.get("output_norm.weight"))?;
        rmsnorm(&x, &head_norm.data, eps, &mut n);
        let lm_head = w
            .get("output.weight")
            .or_else(|_| w.get("token_embd.weight"))?;
        matvec(lm_head, &n, logits_out)?;

        if let Some(h) = next_hidden_out {
            h.copy_from_slice(&x);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Deterministic, non-constant fill so the heads and positions differ.
    fn fill(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.2
            })
            .collect()
    }

    /// Trunk of 1 layer (unused here) + the NextN block at `blk.1`.
    /// `hd` may differ from `d / nh` to exercise `key_length`.
    fn tiny(hd: usize, tied: bool) -> Qwen3_8Mtp {
        let (d, nh, nkv, ff, vocab) = (8usize, 2usize, 1usize, 16usize, 12usize);
        let mut map: HashMap<String, (Vec<f32>, [u32; 4])> = HashMap::new();
        let mut seed = 0u32;
        let mut mat = |rows: usize, cols: usize| {
            seed += 1;
            (fill(rows * cols, seed), [cols as u32, rows as u32, 1, 1])
        };
        let ones = |n: usize| (vec![1.0f32; n], [n as u32, 1, 1, 1]);
        let b = "blk.1";
        map.insert("token_embd.weight".into(), mat(vocab, d));
        if !tied {
            map.insert("output.weight".into(), mat(vocab, d));
        }
        map.insert("output_norm.weight".into(), ones(d));
        map.insert(format!("{b}.nextn.enorm.weight"), ones(d));
        map.insert(format!("{b}.nextn.hnorm.weight"), ones(d));
        map.insert(format!("{b}.nextn.eh_proj.weight"), mat(d, 2 * d));
        map.insert(format!("{b}.attn_norm.weight"), ones(d));
        map.insert(format!("{b}.attn_q.weight"), mat(nh * hd, d));
        map.insert(format!("{b}.attn_k.weight"), mat(nkv * hd, d));
        map.insert(format!("{b}.attn_v.weight"), mat(nkv * hd, d));
        map.insert(format!("{b}.attn_q_norm.weight"), ones(hd));
        map.insert(format!("{b}.attn_k_norm.weight"), ones(hd));
        map.insert(format!("{b}.attn_output.weight"), mat(d, nh * hd));
        map.insert(format!("{b}.ffn_norm.weight"), ones(d));
        map.insert(format!("{b}.ffn_gate.weight"), mat(ff, d));
        map.insert(format!("{b}.ffn_up.weight"), mat(ff, d));
        map.insert(format!("{b}.ffn_down.weight"), mat(d, ff));
        let cfg = ModelConfig {
            arch: "qwen38".into(),
            n_layers: 1,
            n_heads: nh,
            n_kv_heads: nkv,
            key_length: hd,
            d_model: d,
            d_ffn: ff,
            vocab_size: vocab,
            max_seq_len: 64,
            rope_theta: 10_000.0,
            norm_eps: 1e-6,
            n_nextn_predict_layers: 1,
            ..Default::default()
        };
        let mut m = Qwen3_8Mtp::new(cfg);
        m.weights = Some(DenseWeights::from_map(map));
        m
    }

    fn chain(m: &Qwen3_8Mtp, steps: usize) -> Vec<Vec<f32>> {
        let mut h = fill(8, 99);
        let mut tok = 3u32;
        let mut out = Vec::new();
        for pos in 0..steps {
            let mut logits = vec![0f32; 12];
            let mut next = vec![0f32; 8];
            m.forward_step(tok, &h, pos, &mut logits, Some(&mut next))
                .unwrap();
            tok = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap();
            h = next;
            out.push(logits);
        }
        out
    }

    #[test]
    fn chained_steps_are_finite_and_reset_is_exact() {
        let m = tiny(4, false);
        let first = chain(&m, 4);
        assert!(first.iter().flatten().all(|v| v.is_finite()));
        // Later steps attend over earlier rows, so they must differ from step 0.
        assert_ne!(first[0], first[3]);
        m.reset_cache();
        assert_eq!(first, chain(&m, 4));
    }

    #[test]
    fn head_dim_comes_from_key_length() {
        // d_model / n_heads = 4, but the heads are 8 wide.
        let m = tiny(8, false);
        assert!(chain(&m, 2).iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn tied_embeddings_serve_as_lm_head() {
        let m = tiny(4, true);
        assert!(chain(&m, 1)[0].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn qwen38_trunk_is_registered() {
        let cfg = tiny(4, false).config.clone();
        assert!(crate::ArchRegistry::default().build(cfg).is_ok());
    }

    #[test]
    fn rejects_bad_inputs() {
        let m = tiny(4, false);
        let mut logits = vec![0f32; 12];
        assert!(m.forward_step(0, &[0.0; 7], 0, &mut logits, None).is_err());
        assert!(m.forward_step(12, &[0.0; 8], 0, &mut logits, None).is_err());
        assert!(
            m.forward_step(0, &[0.0; 8], 0, &mut [0f32; 11], None)
                .is_err()
        );
        let unloaded = Qwen3_8Mtp::new(m.config.clone());
        assert!(
            unloaded
                .forward_step(0, &[0.0; 8], 0, &mut logits, None)
                .is_err()
        );
    }
}
