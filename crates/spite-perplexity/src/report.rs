//! Human-readable and machine-readable report output.
//!
//! The report is printed to stdout and optionally written as JSON.
//! The JSON format is what `spite-bench` uses when it runs the full
//! PPL + KLD suite and embeds the result in a PR description.

use serde::Serialize;

use crate::kld::KldResult;
use crate::ppl::PplResult;

#[derive(Debug, Serialize)]
pub struct Report {
    pub gpu_arch: String,
    pub model_arch: String,
    pub kernel_path: String,
    pub ppl: Option<PplReport>,
    pub kld: Option<KldReport>,
}

#[derive(Debug, Serialize)]
pub struct PplReport {
    pub ppl: f64,
    pub nll: f64,
    pub n_tokens: usize,
    pub ref_ppl: Option<f64>,
    pub delta_pct: Option<f64>,
    pub passed: bool,
}

#[derive(Debug, Serialize)]
pub struct KldReport {
    pub mean_kld: f64,
    pub max_kld: f64,
    pub p95_kld: f64,
    pub n_positions: usize,
    pub verdict: String,
    pub worst_tokens: Vec<WorstToken>,
}

#[derive(Debug, Serialize)]
pub struct WorstToken {
    pub token_id: u32,
    pub kld: f64,
}

impl Report {
    pub fn new(gpu_arch: &str, model_arch: &str, kernel_path: &str) -> Self {
        Self {
            gpu_arch: gpu_arch.to_owned(),
            model_arch: model_arch.to_owned(),
            kernel_path: kernel_path.to_owned(),
            ppl: None,
            kld: None,
        }
    }

    pub fn with_ppl(mut self, result: &PplResult, reference: Option<&PplResult>) -> Self {
        let (ref_ppl, delta_pct) = match reference {
            Some(r) => {
                let delta = (result.ppl - r.ppl).abs() / r.ppl * 100.0;
                (Some(r.ppl), Some(delta))
            }
            None => (None, None),
        };

        let passed = match (reference, delta_pct) {
            (Some(_), Some(d)) => d < 0.1,
            _ => true,
        };

        self.ppl = Some(PplReport {
            ppl: result.ppl,
            nll: result.nll,
            n_tokens: result.n_tokens,
            ref_ppl,
            delta_pct,
            passed,
        });
        self
    }

    pub fn with_kld(mut self, result: &KldResult) -> Self {
        self.kld = Some(KldReport {
            mean_kld: result.mean_kld,
            max_kld: result.max_kld,
            p95_kld: result.p95_kld,
            n_positions: result.n_positions,
            verdict: result.verdict.to_string(),
            worst_tokens: result
                .worst
                .iter()
                .map(|w| WorstToken {
                    token_id: w.token_id,
                    kld: w.kld,
                })
                .collect(),
        });
        self
    }

    pub fn print(&self) {
        println!("┌─ spite-perplexity report ─────────────────────────────────");
        println!("│ kernel  : {}", self.kernel_path);
        println!("│ model   : {}", self.model_arch);
        println!("│ gpu     : {}", self.gpu_arch);

        if let Some(p) = &self.ppl {
            println!("├─ PPL ─────────────────────────────────────────────────────");
            println!("│ ppl     : {:.4}", p.ppl);
            println!("│ nll     : {:.6}", p.nll);
            println!("│ tokens  : {}", p.n_tokens);
            if let (Some(ref_ppl), Some(delta)) = (p.ref_ppl, p.delta_pct) {
                let marker = if p.passed { "✓" } else { "✗" };
                println!("│ ref ppl : {ref_ppl:.4}   Δ = {delta:.4}%  {marker}");
            }
        }

        if let Some(k) = &self.kld {
            let marker = match k.verdict.as_str() {
                "PASS" => "✓ PASS",
                "WARN" => "⚠ WARN",
                _ => "✗ FAIL",
            };
            println!("├─ KLD ─────────────────────────────────────────────────────");
            println!("│ verdict : {marker}");
            println!("│ mean    : {:.6}", k.mean_kld);
            println!("│ p95     : {:.6}", k.p95_kld);
            println!("│ max     : {:.6}", k.max_kld);
            println!("│ tokens  : {}", k.n_positions);
            if !k.worst_tokens.is_empty() {
                println!("│ worst tokens (token_id → kld):");
                for w in k.worst_tokens.iter().take(5) {
                    println!("│   {:>7}  →  {:.6}", w.token_id, w.kld);
                }
            }
        }

        println!("└───────────────────────────────────────────────────────────");
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Overall pass/fail — both PPL and KLD must pass.
    pub fn passed(&self) -> bool {
        let ppl_ok = self.ppl.as_ref().is_none_or(|p| p.passed);
        let kld_ok = self.kld.as_ref().is_none_or(|k| k.verdict == "PASS");
        ppl_ok && kld_ok
    }
}
