use std::path::Path;
use std::sync::Mutex;

use anyhow::Result;

use spite_dispatch::{DispatchBuilder, KernelSpec};
use spite_executor::{EngineBuilder, ExecutorConfig};
use spite_loader::GgufModel;
use spite_scheduler::Scheduler;

/// Shared server state, held behind Arc<AppState>.
pub struct AppState {
    pub model_arch: String,
    pub gpu_arch: String,
    /// The scheduler owns the Engine and all active request slots.
    /// Mutex because axum handlers run concurrently but inference is serial per GPU.
    pub scheduler: Mutex<Scheduler>,
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

        let _spec = KernelSpec::from_arch(&model_arch, gpu_arch);
        let _dispatch = DispatchBuilder::new(kernels_dir, _spec).build()?;

        let engine = EngineBuilder::new().build(ExecutorConfig::default());
        let scheduler = Scheduler::new(max_concurrent, engine);

        Ok(Self {
            model_arch,
            gpu_arch: gpu_arch.to_owned(),
            scheduler: Mutex::new(scheduler),
        })
    }
}
