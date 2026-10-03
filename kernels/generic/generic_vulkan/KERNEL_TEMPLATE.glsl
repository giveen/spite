// Vulkan compute shader kernel template — generic_vulkan fallback.
//
// This is the fallback for GPUs that have neither CUDA, HIP, nor Metal
// but do have Vulkan 1.3 compute support (Intel Arc, older AMD on Windows,
// etc.). Contributors with these GPUs implement ops here.
//
// Build system: CMakeLists.txt compiles .glsl → SPIR-V with glslangValidator
// or shaderc when SPITE_VULKAN=1 is set and Vulkan SDK is found.
//
// Layout conventions:
//   binding 0 → input tensor A
//   binding 1 → input tensor B (or weight)
//   binding 2 → output tensor C
//   push constants → dimensions (m, k, n) + scale factors
//
// Workgroup size: 8×8×1 is a safe default for portability.
// Tune to 16×16×1 on dedicated GPUs with ≥ 256 invocations/workgroup.

#version 450
#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// ── Push constants ──────────────────────────────────────────────────────────

layout(push_constant) uniform Params {
    uint M;
    uint K;
    uint N;
} params;

// ── Bindings ────────────────────────────────────────────────────────────────

layout(std430, binding = 0) readonly  buffer BufA { float a[]; };
layout(std430, binding = 1) readonly  buffer BufB { float b[]; };
layout(std430, binding = 2) writeonly buffer BufC { float c[]; };

// ── Kernel ──────────────────────────────────────────────────────────────────

void main() {
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    if (row >= params.M || col >= params.N) return;

    float acc = 0.0;
    for (uint k = 0; k < params.K; k++) {
        acc += a[row * params.K + k] * b[k * params.N + col];
    }
    c[row * params.N + col] = acc;
}

// TODO: implement rms_norm.glsl, attention.glsl, ffn.glsl following the same
//       binding and push-constant conventions as this GEMM template.
// TODO: Q4_K dequantization via integer bit manipulation in GLSL.
