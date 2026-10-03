use std::path::Path;

use anyhow::Result;
use tokio::sync::Semaphore;

use spite_dispatch::{DispatchBuilder, DispatchTable};
use spite_loader::GgufModel;

/// Shared server state, held behind Arc<AppState>.
pub struct AppState {
    pub model:    GgufModel,
    pub dispatch: DispatchTable,
    pub gpu_arch: String,
    /// Limits concurrent inference — one token stream at a time on a single GPU.
    pub slots:    Semaphore,
}

impl AppState {
    pub fn load(
        model_path:  &Path,
        kernels_dir: &Path,
        gpu_arch:    &str,
        max_concurrent: usize,
    ) -> Result<Self> {
        let model    = GgufModel::open(model_path)?;
        let dispatch = DispatchBuilder::new(kernels_dir, model.arch(), gpu_arch).build()?;

        Ok(Self {
            model,
            dispatch,
            gpu_arch: gpu_arch.to_owned(),
            slots:    Semaphore::new(max_concurrent),
        })
    }
}
