//! Tensor parallelism for multi-GPU inference.
//!
//! Two independent parallelism strategies:
//!
//! ## Pipeline parallelism
//! Partition the transformer layers across N GPUs.
//! GPU 0 runs layers 0..k, GPU 1 runs k..2k, etc.
//! A micro-batch pipeline keeps all GPUs busy: while GPU 1 computes the
//! second microbatch on layers 0..k, GPU 2 is computing the first microbatch
//! on layers k..2k.
//!
//! ## Tensor parallelism (Megatron-style)
//! Shard the weight matrices column- or row-wise. Each GPU holds 1/N of each
//! weight and computes a partial result; a single all-reduce per layer
//! combines the shards. Good for attention heads (head count ÷ N per GPU)
//! and FFN columns.
//!
//! ## Hybrid
//! Pipeline parallelism across nodes, tensor parallelism within a node.
//! 8-GPU server: 4-way tensor × 2-way pipeline.

pub mod pipeline;
pub mod tensor_par;

/// How many GPUs this process can see.
/// Falls back to 1 if no GPU runtime is present.
pub fn gpu_count() -> usize {
    // TODO: query via CUDA / HIP / Metal
    std::env::var("SPITE_GPU_COUNT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
}

/// Shard strategy to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardStrategy {
    /// Single GPU (no parallelism).
    None,
    /// Layers split across GPUs in sequence.
    Pipeline { n_stages: usize },
    /// Weights split column-wise within each layer.
    Tensor { n_shards: usize },
    /// Tensor within a node, pipeline across nodes.
    Hybrid { n_shards: usize, n_stages: usize },
}

impl Default for ShardStrategy {
    fn default() -> Self {
        let n = gpu_count();
        if n == 1 { Self::None } else { Self::Tensor { n_shards: n } }
    }
}
