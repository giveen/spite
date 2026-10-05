use std::path::Path;
use std::sync::Mutex;

use anyhow::Result;

use spite_dispatch::{DispatchBuilder, KernelSpec};
use spite_executor::{EngineBuilder, Executor, ExecutorConfig};
use spite_loader::GgufModel;
use spite_models::{ArchRegistry, ModelConfig};
use spite_scheduler::Scheduler;
use spite_tokenizer::Tokenizer;

/// Shared server state, held behind Arc<AppState>.
pub struct AppState {
    pub model_arch: String,
    pub gpu_arch: String,
    /// The scheduler owns the Engine and all active request slots.
    /// Mutex because axum handlers run concurrently but inference is serial per GPU.
    pub scheduler: Mutex<Scheduler>,
    pub tokenizer: Tokenizer,
    /// Direct executor for the chat path (single-GPU serial inference).
    pub executor: Mutex<Executor>,
}

impl AppState {
    pub fn load(
        model_path: &Path,
        kernels_dir: &Path,
        gpu_arch: &str,
        max_concurrent: usize,
    ) -> Result<Self> {
        let model = GgufModel::open(model_path)?;
        let model_arch = model.arch().to_owned();

        let mut spec = KernelSpec::from_arch(&model_arch, gpu_arch);
        if spec.card_id.is_empty() {
            spec.card_id = spite_dispatch::detect_card_id("");
        }
        let _dispatch = DispatchBuilder::new(kernels_dir, spec).build()?;

        let hp = spite_loader::config::ModelHyperparams::from_gguf(&model);
        let mut arch = ArchRegistry::default().build(ModelConfig::from(hp))?;
        arch.load_weights(&model)?;
        let tokenizer = Tokenizer::from_gguf(&model)?;

        let engine = EngineBuilder::new().build(ExecutorConfig::default());
        let scheduler = Scheduler::new(max_concurrent, engine);

        let mut executor = Executor::new(ExecutorConfig::default());
        executor.load_model(arch);

        Ok(Self {
            model_arch,
            gpu_arch: gpu_arch.to_owned(),
            scheduler: Mutex::new(scheduler),
            tokenizer,
            executor: Mutex::new(executor),
        })
    }
}
