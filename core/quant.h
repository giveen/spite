/*
 * spite/core/quant.h — GGUF quantization block layouts + reference dequant.
 *
 * Block structs (block_q4_K, ...) come from core/ggml-common.h, vendored
 * verbatim from llama.cpp (MIT, see core/THIRD_PARTY.md).
 */

#pragma once
#include <stdint.h>
#include "abi.h"

#ifndef GGML_COMMON_DECL_C
#define GGML_COMMON_DECL_C
#endif
#include "ggml-common.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Dequantize n elements of `type` (n/block_elements blocks) into dst (fp32).
 * Handles F32/F16/BF16 and every GGUF quant type in SpiteType.
 * Returns 0 on success; -1 if the type is unsupported or n is not a
 * multiple of spite_type_block_elements(type).
 */
int spite_dequantize_row(SpiteType type, const void *src, float *dst, int64_t n);

#ifdef __cplusplus
}
#endif
