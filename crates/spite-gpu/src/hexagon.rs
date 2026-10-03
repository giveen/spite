//! Hexagon backend — Qualcomm Snapdragon DSP / HTP / HTA.
//!
//! Targets the dedicated AI acceleration silicon inside Snapdragon SoCs,
//! distinct from the Adreno GPU (which uses the OpenCL backend).
//!
//! Relevant blocks:
//! - **Hexagon HTP** (High-Throughput Processor): INT8/INT16 tensor operations,
//!   800+ GOPS on Snapdragon 8 Gen 3. Primary target for on-device LLM.
//! - **Hexagon HTA** (High-Throughput Accelerator): dedicated matmul engine,
//!   Snapdragon 8 Elite and X Elite SoCs (2024+).
//! - Snapdragon X Elite / X Plus laptops (Oryon CPU + Hexagon NPU + Adreno GPU).
//!
//! # Build requirements
//!
//! - Qualcomm AI Engine Direct (QNN SDK) or Hexagon SDK
//! - `SPITE_HEXAGON=1` CMake flag
//! - Android NDK for cross-compilation (arm64-v8a target)
//! - HTP .dlc graph compilation step (offline, analogous to TensorRT engine build)
//!
//! # Programming model
//!
//! Unlike CUDA/HIP, Hexagon HTP is accessed via:
//! 1. **QNN API**: graph-based (build op graph offline → execute at runtime).
//!    Most similar to TensorRT or CoreML.
//! 2. **HexagonNN**: lower-level direct DSP invocation.
//! 3. **SNPE** (Snapdragon Neural Processing Engine): higher-level SDK.
//!
//! Spite should target QNN for maximum portability across SoC generations.
//!
//! # Quantization on Hexagon
//!
//! HTP natively accelerates INT8 and INT4 (Snapdragon 8 Elite). F16 runs
//! but is slower. Map spite's Q4_0/Q8_0 blocks → QNN quantized tensors.

use crate::GpuError;

pub fn alloc(_size: usize) -> Result<*mut u8, GpuError> {
    // TODO: QNN / rpcmem_alloc for DMA-accessible memory
    // rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size)
    Err(GpuError::BackendUnavailable(crate::GpuBackend::Hexagon))
}

pub fn free(_ptr: *mut u8) {
    // TODO: rpcmem_free(ptr)
}

pub fn upload(_ptr: *mut u8, _src: &[u8]) -> Result<(), GpuError> {
    // Hexagon shares physical memory with the ARM CPU on-chip.
    // "Upload" is a cache flush + DSP pointer registration, not a DMA copy.
    // TODO: __attribute__((aligned(128))) memcpy then cache invalidation
    Err(GpuError::BackendUnavailable(crate::GpuBackend::Hexagon))
}

pub fn download(_ptr: *mut u8, _dst: &mut [u8]) -> Result<(), GpuError> {
    // Cache invalidation on ARM side after DSP write.
    Err(GpuError::BackendUnavailable(crate::GpuBackend::Hexagon))
}
