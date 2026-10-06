# Third-party code

## ggml-common.h (and dequantize_row_* bodies in quant.c)

- Source: llama.cpp `ggml/src/ggml-common.h`, `ggml/src/ggml-quants.c`
- Repository: https://github.com/ggml-org/llama.cpp
- Commit: a25c9865fe03c954c93fd755b5d79ae86ba99750
- `ggml-common.h` is a verbatim copy.
- `core/gpu/quant_dequant.h` re-derives the per-type decode order of ggml-cuda
  `dequantize.cuh` / `convert.cu` (same file, same commit); bit-exactness against
  `spite_dequantize_row()` is proven by `tools/verify/quant_gpu_test.cu`.

MIT License — Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
