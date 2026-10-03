/*
 * spite/dispatch/dispatch.h  —  runtime kernel selection
 *
 * Detects the GPU at startup, walks the kernels/ directory tree,
 * and builds a dispatch table: for each (model_arch, quant_type, op)
 * triple, which kernel implementation wins.
 *
 * Fallback chain:
 *   kernels/<model>/<exact_gpu_arch>/  (e.g. sm_89)
 *   kernels/<model>/generic_cuda/
 *   kernels/<model>/generic/
 *   kernels/generic/<exact_gpu_arch>/
 *   kernels/generic/generic/           (always present, always correct)
 */

#pragma once
#include "../core/abi.h"

typedef struct {
    char model_arch[64];
    char gpu_arch[64];          /* detected at runtime, e.g. "sm_89" */
} spite_dispatch_config_t;

typedef struct {
    spite_rms_norm_fn  rms_norm;
    spite_attention_fn attention;
    spite_ffn_fn       ffn;
    spite_layer_fn     layer;
} spite_dispatch_table_t;

/*
 * Build the dispatch table for the given model and detected GPU.
 * kernels_dir: path to the kernels/ directory.
 * Returns 0 on success. Always succeeds — falls back to generic.
 */
int spite_dispatch_build(
    spite_dispatch_table_t       *table,
    const spite_dispatch_config_t *cfg,
    const char                    *kernels_dir
);

/* Print which kernel won each slot and why (for debugging / --verbose) */
void spite_dispatch_print(const spite_dispatch_table_t *table);

/* Detect the current GPU and fill cfg->gpu_arch */
int spite_detect_gpu(spite_dispatch_config_t *cfg);
