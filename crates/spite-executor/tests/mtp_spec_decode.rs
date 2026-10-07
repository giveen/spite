//! Executor-level MTP self-speculative decoding on the CPU backend.
//!
//! The tiny NextN fixture (`tiny-qwen35-mtp-full-f16.gguf`) is pinned to
//! llama.cpp. Speculative decoding must be **exact** for greedy sampling: the
//! MTP head only proposes, and every emitted token is the trunk's argmax, so
//! the token stream must equal a plain greedy decode. A regression here means
//! the bench's "acceptance" was measuring the wrong distribution.

use std::path::{Path, PathBuf};
use std::process::Command;

use spite_executor::{Executor, ExecutorConfig, SpecDecodeConfig};
use spite_loader::GgufModel;
use spite_models::ModelConfig;
use spite_models::hybrid::{HybridDecoder, LayerSplit};

/// Compile the generic kernel into `<tmp>/generic/generic/libkernel_generic.so`.
fn build_generic_kernel(tag: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let kdir = std::env::temp_dir().join(format!("spite-mtp-spec-{tag}-{}", std::process::id()));
    let out_dir = kdir.join("generic").join("generic");
    std::fs::create_dir_all(&out_dir).expect("create kernel dir");
    let mut cmd = Command::new(std::env::var("CC").unwrap_or_else(|_| "cc".into()));
    cmd.args(["-std=c11", "-O2", "-fPIC", "-shared"])
        .arg(format!("-I{}", root.display()))
        .arg(root.join("core/quant.c"));
    for f in ["kernel.c", "ops.c", "linear_attn.c"] {
        cmd.arg(root.join("kernels/generic/generic").join(f));
    }
    cmd.arg("-lm")
        .arg("-o")
        .arg(out_dir.join("libkernel_generic.so"));
    assert!(
        cmd.status().expect("run C compiler").success(),
        "generic kernel failed to compile"
    );
    kdir
}

fn load() -> (HybridDecoder, GgufModel, PathBuf) {
    let kdir = build_generic_kernel("exec");
    let file = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../spite-models/tests/data/tiny-qwen35-mtp-full-f16.gguf");
    let gguf = GgufModel::open(&file).expect("open fixture");
    let cfg = ModelConfig::from(spite_loader::config::ModelHyperparams::from_gguf(&gguf));
    let (table, backend) = HybridDecoder::resolve_table(&cfg.arch, "generic", &kdir, false)
        .expect("generic kernels provide every hybrid op");
    let (model, report) =
        HybridDecoder::load_split(cfg, &gguf, table, backend, 32, &LayerSplit::default())
            .expect("load hybrid decoder");
    assert!(
        report.mtp && model.has_mtp(),
        "fixture must carry a NextN head"
    );
    (model, gguf, kdir)
}

fn ids(pieces: &[(u32, String)]) -> Vec<u32> {
    pieces.iter().map(|(id, _)| *id).collect()
}

#[test]
fn speculative_greedy_matches_plain_greedy() {
    use spite_tokenizer::Tokenizer;
    let (model, gguf, kdir) = load();
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let prompt = tokenizer.encode("AB", true).expect("encode");

    let mut exec = Executor::new(ExecutorConfig::default());
    exec.load_model(Box::new(model));
    let plain = exec
        .generate(&tokenizer, &prompt, 12, 0.0, 7)
        .expect("plain");
    exec.reset();
    let (spec, stats) = exec
        .generate_speculative_with_stats(
            &tokenizer,
            &prompt,
            12,
            SpecDecodeConfig {
                temperature: 0.0,
                seed: 7,
                n_draft: 3,
            },
        )
        .expect("speculative");

    assert!(!plain.is_empty(), "fixture produced no tokens");
    assert!(
        stats.drafted > 0,
        "the MTP head never ran: {stats:?} (bench acceptance would be a false zero)"
    );
    assert_eq!(
        ids(&spec),
        ids(&plain),
        "MTP speculative greedy must equal plain greedy"
    );

    let _ = std::fs::remove_dir_all(kdir);
}
