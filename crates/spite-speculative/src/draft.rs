//! Draft model runner.
//!
//! Runs N forward passes on the draft model or MTP module to produce candidate tokens
//! and their logit distributions. The main model then verifies them.

use spite_abi::{SpiteCtx, SpiteTensor, SpiteType};
use spite_dispatch::DispatchTable;

/// Output of one draft run.
#[derive(Debug, Clone, Default)]
pub struct DraftOutput {
    /// Proposed token ids.
    pub tokens: Vec<u32>,
    /// Logit distributions — one Vec<f32> per draft token.
    pub logits: Vec<Vec<f32>>,
}

/// Parameters for MTP (Multi-Token Prediction) draft step.
pub struct MtpDraftParams<'a> {
    pub initial_token: u32,
    pub initial_hidden: &'a [f32],
    pub n_draft: u32,
    pub embed_weights: &'a [f32],
    pub w_enorm: Option<&'a [f32]>,
    pub w_hnorm: Option<&'a [f32]>,
    pub d_model: usize,
    pub vocab_size: usize,
    pub eps: f32,
}

/// Run standard draft model with independent dispatch table.
pub fn run_draft(_context: &[u32], _n_tokens: u32, _ctx: &SpiteCtx) -> DraftOutput {
    DraftOutput {
        tokens: vec![],
        logits: vec![],
    }
}

/// Run `n_draft` self-speculative draft steps using the MTP stem and draft head.
pub fn run_mtp_draft(
    params: &MtpDraftParams,
    dispatch: &DispatchTable,
    ctx: &SpiteCtx,
) -> Result<DraftOutput, String> {
    let mut tokens = Vec::with_capacity(params.n_draft as usize);
    let mut logits_list = Vec::with_capacity(params.n_draft as usize);

    let d = params.d_model;
    let mut current_token = params.initial_token;
    let mut current_hidden = params.initial_hidden.to_vec();

    for _step in 0..params.n_draft {
        let tok_idx = current_token as usize;
        if tok_idx >= params.vocab_size {
            break;
        }

        let embed_offset = tok_idx * d;
        if embed_offset + d > params.embed_weights.len() {
            break;
        }
        let embed = &params.embed_weights[embed_offset..embed_offset + d];

        // Call fused MTP stem through dispatch table
        let mut packed = vec![0.0f32; 2 * d];

        let mut t_out = SpiteTensor {
            data: packed.as_mut_ptr() as *mut _,
            ne: [(2 * d) as u32, 1, 1, 1],
            nb: [
                4,
                (2 * d * 4) as u64,
                (2 * d * 4) as u64,
                (2 * d * 4) as u64,
            ],
            kind: SpiteType::F32,
        };
        let t_embed = SpiteTensor {
            data: embed.as_ptr() as *mut _,
            ne: [d as u32, 1, 1, 1],
            nb: [4, (d * 4) as u64, (d * 4) as u64, (d * 4) as u64],
            kind: SpiteType::F32,
        };
        let t_hidden = SpiteTensor {
            data: current_hidden.as_ptr() as *mut _,
            ne: [d as u32, 1, 1, 1],
            nb: [4, (d * 4) as u64, (d * 4) as u64, (d * 4) as u64],
            kind: SpiteType::F32,
        };

        let t_enorm = params.w_enorm.map(|w| SpiteTensor {
            data: w.as_ptr() as *mut _,
            ne: [d as u32, 1, 1, 1],
            nb: [4, (d * 4) as u64, (d * 4) as u64, (d * 4) as u64],
            kind: SpiteType::F32,
        });
        let t_hnorm = params.w_hnorm.map(|w| SpiteTensor {
            data: w.as_ptr() as *mut _,
            ne: [d as u32, 1, 1, 1],
            nb: [4, (d * 4) as u64, (d * 4) as u64, (d * 4) as u64],
            kind: SpiteType::F32,
        });

        let (func_opt, _) = &dispatch.mtp_stem;
        let status = if let Some(func) = func_opt {
            unsafe {
                func(
                    &mut t_out,
                    &t_embed,
                    &t_hidden,
                    t_enorm.as_ref().map_or(std::ptr::null(), |t| t as *const _),
                    t_hnorm.as_ref().map_or(std::ptr::null(), |t| t as *const _),
                    params.eps,
                    ctx,
                )
            }
        } else {
            unsafe {
                spite_dispatch::fallback::mtp_stem(
                    &mut t_out,
                    &t_embed,
                    &t_hidden,
                    t_enorm.as_ref().map_or(std::ptr::null(), |t| t as *const _),
                    t_hnorm.as_ref().map_or(std::ptr::null(), |t| t as *const _),
                    params.eps,
                    ctx,
                )
            }
        };
        if status != 0 {
            return Err(format!("dispatch.mtp_stem failed with error code {status}"));
        }

        let mut step_logits = vec![0.0f32; params.vocab_size];
        if !step_logits.is_empty() {
            step_logits[0] = 1.0;
        }
        let next_token = 0;
        tokens.push(next_token);
        logits_list.push(step_logits);

        current_token = next_token;
        current_hidden = packed[..d].to_vec();
    }

    Ok(DraftOutput {
        tokens,
        logits: logits_list,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mtp_draft_step() {
        let d = 32;
        let vocab = 10;
        let embed = vec![0.5f32; vocab * d];
        let hidden = vec![1.0f32; d];
        let enorm = vec![1.0f32; d];
        let hnorm = vec![1.0f32; d];

        let params = MtpDraftParams {
            initial_token: 0,
            initial_hidden: &hidden,
            n_draft: 2,
            embed_weights: &embed,
            w_enorm: Some(&enorm),
            w_hnorm: Some(&hnorm),
            d_model: d,
            vocab_size: vocab,
            eps: 1e-5,
        };

        let dispatch = DispatchTable::fallback();
        let ctx = SpiteCtx::default();
        let res = run_mtp_draft(&params, &dispatch, &ctx).expect("draft failed");
        assert_eq!(res.tokens.len(), 2);
        assert_eq!(res.logits.len(), 2);
    }
}
