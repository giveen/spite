/*
 * spite/loader/gguf.h  —  GGUF file loader
 *
 * Reads a GGUF file, mmaps the tensor data, and exposes tensors
 * by name. The model arch and hyperparameters live in the KV store.
 *
 * No copies are made — tensors point directly into the mmap'd buffer.
 * The model stays resident as long as spite_gguf_t is alive.
 */

#pragma once
#include <stdint.h>
#include <stddef.h>
#include "../core/abi.h"

typedef struct spite_gguf spite_gguf_t;

/* Open and mmap a GGUF file. Returns NULL on error (check errno). */
spite_gguf_t *spite_gguf_open(const char *path);

void spite_gguf_close(spite_gguf_t *g);

/* ── Metadata ─────────────────────────────────────────────────────────
 * GGUF KV store access. Keys are dotted strings like
 * "llama.context_length", "general.architecture".
 */
const char    *spite_gguf_get_str (const spite_gguf_t *g, const char *key);
uint32_t       spite_gguf_get_u32 (const spite_gguf_t *g, const char *key, uint32_t def);
float          spite_gguf_get_f32 (const spite_gguf_t *g, const char *key, float def);

/* The model architecture string, e.g. "llama", "mistral", "phi3" */
const char    *spite_gguf_arch(const spite_gguf_t *g);

/* ── Tensor access ────────────────────────────────────────────────────
 * Tensor names follow the GGUF convention:
 *   "blk.0.attn_q.weight"
 *   "blk.0.ffn_gate.weight"
 *   "output_norm.weight"
 *   "token_embd.weight"
 *
 * Returns a tensor whose .data points into the mmap'd buffer.
 * Returns a zero tensor (data=NULL) if the name doesn't exist.
 */
spite_tensor_t spite_gguf_tensor(const spite_gguf_t *g, const char *name);

/* Total number of tensors in the file */
int spite_gguf_n_tensors(const spite_gguf_t *g);

/* Iterate tensors by index (for tooling / debugging) */
const char    *spite_gguf_tensor_name(const spite_gguf_t *g, int i);
spite_tensor_t spite_gguf_tensor_by_index(const spite_gguf_t *g, int i);
