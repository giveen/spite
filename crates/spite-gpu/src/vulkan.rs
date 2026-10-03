//! Vulkan compute backend — GPU-vendor-agnostic portability layer.
//!
//! Targets GPUs that have neither CUDA nor HIP nor Metal but do have
//! Vulkan 1.3 compute support (Intel Arc, older AMD on Windows, etc.).
//! Also useful as a testing backend when no dedicated GPU is present.
//!
//! Key objects (to be managed here):
//!   VkInstance, VkDevice, VkQueue (compute), VkCommandPool,
//!   VkDescriptorPool for binding weight buffers to shader pipelines.

use crate::GpuError;

pub fn alloc(_size: usize) -> Result<*mut u8, GpuError> {
    // TODO: vkAllocateMemory(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    //       vkCreateBuffer, vkBindBufferMemory
    Err(GpuError::BackendUnavailable(crate::GpuBackend::Vulkan))
}

pub fn free(_ptr: *mut u8) {
    // TODO: vkFreeMemory / vkDestroyBuffer
}

pub fn upload(_dst: *mut u8, _src: &[u8]) -> Result<(), GpuError> {
    // TODO: staging buffer → vkCmdCopyBuffer → device buffer
    Ok(())
}

pub fn download(_src: *mut u8, _dst: &mut [u8]) -> Result<(), GpuError> {
    // TODO: device buffer → staging buffer → host memcpy
    Ok(())
}
