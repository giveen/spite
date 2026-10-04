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

// ShardStrategy lives in spite-abi to avoid a dependency cycle.
pub use spite_abi::ShardStrategy;

/// How many GPUs this process can see.
/// Falls back to 1 if no GPU runtime is present.
pub fn gpu_count() -> usize {
    // TODO: query via CUDA / HIP / Metal
    std::env::var("SPITE_GPU_COUNT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
}

/// Returns the best `ShardStrategy` for the detected GPU count.
/// Use this instead of `ShardStrategy::default()` when you want
/// the runtime to automatically pick tensor parallelism on multi-GPU boxes.
pub fn gpu_aware_default() -> ShardStrategy {
    let n = gpu_count();
    if n == 1 { ShardStrategy::None } else { ShardStrategy::Tensor { n_shards: n } }
}
