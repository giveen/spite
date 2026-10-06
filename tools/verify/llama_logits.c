/*
 * tools/verify/llama_logits.c - llama.cpp reference logits for a token
 * sequence.
 *
 * Feeds raw token ids (no tokenizer involved) through llama.cpp, one token per
 * decode call like spite's forward, and writes the logits of every position to
 * a raw little-endian f32 file [n_tokens, n_vocab]. Compare with spite using
 * tools/verify/compare_logits.py.
 *
 * Build (from repo root; llama.cpp built in /mnt/storage/llama.cpp/build):
 *   L=/mnt/storage/llama.cpp
 *   cc -O2 -Wall -Wextra -I$L/include -I$L/ggml/include
 * tools/verify/llama_logits.c \ -L$L/build/bin -lllama -lggml -lggml-base
 * -Wl,-rpath,$L/build/bin -o /tmp/llama_logits Run: /tmp/llama_logits
 * MODEL.gguf OUT.f32 NGL TOK0 TOK1 ...
 */
#include "llama.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s model.gguf out.f32 n_gpu_layers tok...\n",
            argv[0]);
    return 2;
  }
  const int n_tok = argc - 4;
  llama_backend_init();
  struct llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = atoi(argv[3]);
  struct llama_model *model = llama_model_load_from_file(argv[1], mp);
  if (!model) {
    fprintf(stderr, "model load failed\n");
    return 1;
  }
  struct llama_context_params cp = llama_context_default_params();
  cp.n_ctx = (uint32_t)(n_tok + 8);
  cp.n_batch = 1;
  cp.n_ubatch = 1;
  struct llama_context *ctx = llama_init_from_model(model, cp);
  if (!ctx) {
    fprintf(stderr, "context init failed\n");
    return 1;
  }
  const struct llama_vocab *vocab = llama_model_get_vocab(model);
  const int n_vocab = llama_vocab_n_tokens(vocab);
  FILE *out = fopen(argv[2], "wb");
  if (!out) {
    perror("open out");
    return 1;
  }
  for (int i = 0; i < n_tok; i++) {
    llama_token t = (llama_token)atoi(argv[4 + i]);
    struct llama_batch b = llama_batch_get_one(&t, 1);
    if (llama_decode(ctx, b) != 0) {
      fprintf(stderr, "decode failed at %d\n", i);
      return 1;
    }
    const float *lg = llama_get_logits_ith(ctx, -1);
    if (!lg) {
      fprintf(stderr, "no logits at %d\n", i);
      return 1;
    }
    fwrite(lg, sizeof(float), (size_t)n_vocab, out);
  }
  fclose(out);
  fprintf(stderr, "wrote %d positions x %d vocab\n", n_tok, n_vocab);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
