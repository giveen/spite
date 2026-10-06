/*
 * tools/verify/quant_gpu_test.cu — device-level proof for
 * core/gpu/quant_*.h.
 *
 * For every SpiteType (28) this checks, on the GPU:
 *   1. decode: sq::<Type>::decode() over random packed blocks is BIT-EXACT vs
 *      the CPU reference spite_dequantize_row() (core/quant.c, itself proven
 *      bit-exact vs ggml by quant_oracle.c).  Pass A patches every fp16/E8M0
 *      scale to a finite value (as in quant_oracle.c); pass B keeps raw random
 *      bytes (NaN/Inf scales included; NaN==NaN, any payload, is accepted).
 *   2. gemv: sq::gemv() (F32 activations, F32 accumulation) vs a double-
 *      precision host dot of the dequantized row; overwrite, accumulate, an
 *      unaligned x, a one-block row and an odd block count.  The reported error
 *      is |gpu - ref| / sum|w_i x_i| (relative to the dot's own conditioning;
 *      plain |gpu-ref|/|ref| explodes whenever random signs cancel to ~0); it
 *      must be <= 1e-4.  Max absolute error is printed too.
 *   3. rejects: cols not a multiple of the block size, unknown type, misaligned
 *      weights all return -1.
 *
 * Build (from repo root; same flags as CMakeLists.txt uses for kernels):
 *   gcc -std=c11 -O2 -Wall -Wextra -Icore -c core/quant.c -o /tmp/spite_quant.o
 *   nvcc -std=c++20 -O2 --use_fast_math --expt-relaxed-constexpr -arch=sm_120
 * -I. \ tools/verify/quant_gpu_test.cu /tmp/spite_quant.o -o
 * /tmp/quant_gpu_test Run: /tmp/quant_gpu_test                     # all
 * checks, exit 1 on any failure /tmp/quant_gpu_test --bench             # +
 * GEMV GB/s, 14336x4096, 5 types /tmp/quant_gpu_test --bench-all         # +
 * GEMV GB/s for all 28 types
 *
 * Scales are patched to finite values; MXFP4's E8M0 byte is kept >= 2 because
 * --use_fast_math (ftz) flushes the float denormals that bytes 0/1 denote.
 */
#include "core/gpu/quant_gemv.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

extern "C" int spite_dequantize_row(SpiteType type, const void *src, float *dst,
                                    int64_t n);

namespace {

struct TypeInfo {
  SpiteType t;
  const char *name;
};
const TypeInfo ALL[] = {
    {SPITE_TYPE_F32, "F32"},         {SPITE_TYPE_F16, "F16"},
    {SPITE_TYPE_BF16, "BF16"},       {SPITE_TYPE_Q4_0, "Q4_0"},
    {SPITE_TYPE_Q4_1, "Q4_1"},       {SPITE_TYPE_Q5_0, "Q5_0"},
    {SPITE_TYPE_Q5_1, "Q5_1"},       {SPITE_TYPE_Q8_0, "Q8_0"},
    {SPITE_TYPE_Q1_0, "Q1_0"},       {SPITE_TYPE_Q2_0, "Q2_0"},
    {SPITE_TYPE_Q2_K, "Q2_K"},       {SPITE_TYPE_Q3_K, "Q3_K"},
    {SPITE_TYPE_Q4_K, "Q4_K"},       {SPITE_TYPE_Q5_K, "Q5_K"},
    {SPITE_TYPE_Q6_K, "Q6_K"},       {SPITE_TYPE_IQ2_XXS, "IQ2_XXS"},
    {SPITE_TYPE_IQ2_XS, "IQ2_XS"},   {SPITE_TYPE_IQ2_S, "IQ2_S"},
    {SPITE_TYPE_IQ3_XXS, "IQ3_XXS"}, {SPITE_TYPE_IQ3_S, "IQ3_S"},
    {SPITE_TYPE_IQ1_S, "IQ1_S"},     {SPITE_TYPE_IQ1_M, "IQ1_M"},
    {SPITE_TYPE_IQ4_NL, "IQ4_NL"},   {SPITE_TYPE_IQ4_XS, "IQ4_XS"},
    {SPITE_TYPE_TQ1_0, "TQ1_0"},     {SPITE_TYPE_TQ2_0, "TQ2_0"},
    {SPITE_TYPE_MXFP4, "MXFP4"},     {SPITE_TYPE_NVFP4, "NVFP4"},
};
constexpr int NTYPES = sizeof ALL / sizeof *ALL;
static_assert(NTYPES == 28, "all 28 SpiteTypes");

void ck(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
    exit(2);
  }
}

// ── seeded data + scale patching (same idea as quant_oracle.c) ────────────

uint64_t rng = 0x9E3779B97F4A7C15ull;
uint32_t rnd() {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return static_cast<uint32_t>(rng >> 16);
}
float frnd() {
  return static_cast<float>(rnd() % 20001) / 10000.0f - 1.0f;
} /* [-1, 1] */

/* finite half with exponent field 8..16 (|v| in 2^-7 .. 2^2) */
uint16_t sane_half() {
  return static_cast<uint16_t>((rnd() & 0x8000) | ((8 + rnd() % 9) << 10) |
                               (rnd() & 0x3FF));
}
void put16(uint8_t *b, size_t off) {
  uint16_t h = sane_half();
  memcpy(b + off, &h, 2);
}

/* Overwrite every scale field of one block with a finite value.  narrow: keep
 * MXFP4 scales near 1 so GEMV sums stay far from float overflow. */
void patch(SpiteType t, uint8_t *b, bool narrow) {
  switch (t) {
  case SPITE_TYPE_F32: {
    float f = static_cast<float>(static_cast<int>(rnd() % 2001) - 1000) / 7.0f;
    memcpy(b, &f, 4);
    break;
  }
  case SPITE_TYPE_F16:
    put16(b, 0);
    break;
  case SPITE_TYPE_BF16:
    b[1] = static_cast<uint8_t>((rnd() & 0x80) | (0x30 + rnd() % 0x20));
    break;
  case SPITE_TYPE_Q4_0:
  case SPITE_TYPE_Q5_0:
  case SPITE_TYPE_Q8_0:
  case SPITE_TYPE_Q1_0:
  case SPITE_TYPE_Q2_0:
  case SPITE_TYPE_IQ4_NL:
    put16(b, 0);
    break;
  case SPITE_TYPE_Q4_1:
  case SPITE_TYPE_Q5_1:
    put16(b, 0);
    put16(b, 2);
    break;
  case SPITE_TYPE_Q2_K:
    put16(b, 80);
    put16(b, 82);
    break;
  case SPITE_TYPE_Q3_K:
    put16(b, 108);
    break;
  case SPITE_TYPE_Q4_K:
  case SPITE_TYPE_Q5_K:
    put16(b, 0);
    put16(b, 2);
    break;
  case SPITE_TYPE_Q6_K:
    put16(b, 208);
    break;
  case SPITE_TYPE_IQ2_XXS:
  case SPITE_TYPE_IQ2_XS:
  case SPITE_TYPE_IQ2_S:
  case SPITE_TYPE_IQ3_XXS:
  case SPITE_TYPE_IQ3_S:
  case SPITE_TYPE_IQ1_S:
  case SPITE_TYPE_IQ4_XS:
    put16(b, 0);
    break;
  case SPITE_TYPE_IQ1_M: {
    /* fp16 scale is spread over the top nibble of each of 4 uint16 scales
     * (offset 48) */
    uint16_t h = sane_half(), sc[4];
    memcpy(sc, b + 48, 8);
    for (int i = 0; i < 4; i++)
      sc[i] = static_cast<uint16_t>((sc[i] & 0x0FFF) |
                                    (((h >> (4 * i)) & 0xF) << 12));
    memcpy(b + 48, sc, 8);
    break;
  }
  case SPITE_TYPE_TQ1_0:
    put16(b, 52);
    break;
  case SPITE_TYPE_TQ2_0:
    put16(b, 64);
    break;
  case SPITE_TYPE_MXFP4:
    b[0] = static_cast<uint8_t>(narrow ? 110 + rnd() % 30 : 2 + rnd() % 250);
    break;
  case SPITE_TYPE_NVFP4:
    break; /* UE4M3: every byte is finite */
  default:
    break;
  }
}

void fill(SpiteType t, std::vector<uint8_t> &buf, size_t nblk,
          bool patch_scales, bool narrow) {
  const size_t sb = spite_type_block_bytes(t);
  buf.resize(nblk * sb);
  for (auto &c : buf)
    c = static_cast<uint8_t>(rnd());
  for (size_t k = 0; k < nblk; k++) {
    if (patch_scales)
      patch(t, buf.data() + k * sb, narrow);
    else if (t == SPITE_TYPE_MXFP4)
      buf[k * sb] = static_cast<uint8_t>(2 + buf[k * sb] % 250);
  }
}

// ── device kernels under test ─────────────────────────────────────────────

template <class Q>
__global__ void decode_kernel(const uint8_t *src, float *dst, int nblk) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nblk * Q::NS)
    return;
  const int b = i / Q::NS, s = i % Q::NS;
  float v[Q::NV];
  Q::decode(src + static_cast<size_t>(b) * Q::BYTES, s, v);
#pragma unroll
  for (int g = 0; g < Q::NG; ++g)
#pragma unroll
    for (int k = 0; k < 4; ++k)
      dst[static_cast<size_t>(b) * Q::QK + Q::start(s, g) + k] = v[4 * g + k];
}

template <class Q>
__global__ void decode_dense_kernel(const uint8_t *src, float *dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = Q::load(src, i);
}

/* decode n elements (n % block elements == 0) of `t` on the device */
int decode_gpu(SpiteType t, const uint8_t *d_src, float *d_dst, int n) {
  return sq::visit_type(t, [&](auto tag) -> int {
    using Q = typename decltype(tag)::type;
    if constexpr (Q::DENSE) {
      decode_dense_kernel<Q><<<(n + 255) / 256, 256>>>(d_src, d_dst, n);
    } else {
      const int nblk = n / Q::QK;
      decode_kernel<Q><<<(nblk * Q::NS + 255) / 256, 256>>>(d_src, d_dst, nblk);
    }
    return 0;
  });
}

// ── checks ────────────────────────────────────────────────────────────────

bool same_bits(const float *a, const float *b, int64_t n, bool allow_nan,
               int64_t *bad) {
  for (int64_t i = 0; i < n; i++) {
    if (memcmp(a + i, b + i, 4) == 0)
      continue;
    if (allow_nan && isnan(a[i]) && isnan(b[i]))
      continue;
    *bad = i;
    return false;
  }
  return true;
}

/* returns an error string or nullptr */
const char *check_decode(SpiteType t, bool raw) {
  const uint64_t sb = spite_type_block_bytes(t),
                 se = spite_type_block_elements(t);
  const int n = 16384; /* 64 K-blocks / 512 32-blocks / 128 Q1_0 blocks / 16384
                          dense elements */
  const size_t nblk = n / se;
  std::vector<uint8_t> src;
  fill(t, src, nblk, !raw, false);
  std::vector<float> ref(n), got(n);
  if (spite_dequantize_row(t, src.data(), ref.data(), n) != 0)
    return "spite_dequantize_row failed";
  uint8_t *d_src;
  float *d_dst;
  ck(cudaMalloc(&d_src, src.size()), "malloc src");
  ck(cudaMalloc(&d_dst, n * 4), "malloc dst");
  ck(cudaMemcpy(d_src, src.data(), src.size(), cudaMemcpyHostToDevice), "h2d");
  ck(cudaMemset(d_dst, 0xFF, n * 4),
     "memset"); /* NaN: an uncovered element can't match */
  const int rc = decode_gpu(t, d_src, d_dst, n);
  ck(cudaGetLastError(), "decode launch");
  ck(cudaMemcpy(got.data(), d_dst, n * 4, cudaMemcpyDeviceToHost), "d2h");
  cudaFree(d_src);
  cudaFree(d_dst);
  (void)sb;
  if (rc)
    return "visit_type rejected the type";
  int64_t bad = -1;
  if (!same_bits(got.data(), ref.data(), n, raw, &bad)) {
    static char msg[160];
    snprintf(
        msg, sizeof msg,
        "decode mismatch at element %lld: gpu %.9g (0x%08x) ref %.9g (0x%08x)",
        static_cast<long long>(bad), got[bad],
        *reinterpret_cast<uint32_t *>(&got[bad]), ref[bad],
        *reinterpret_cast<uint32_t *>(&ref[bad]));
    return msg;
  }
  return nullptr;
}

struct GemvErr {
  double max_abs = 0, max_rel = 0;
};

/* One GEMV configuration; err accumulates the worst case.  Returns error string
 * or nullptr. */
const char *check_gemv(SpiteType t, int rows, int cols, bool accumulate,
                       bool unaligned_x, GemvErr &err) {
  const uint64_t sb = spite_type_block_bytes(t),
                 se = spite_type_block_elements(t);
  const size_t nb = cols / se, row_bytes = sb * nb;
  std::vector<uint8_t> w;
  fill(t, w, nb * rows, true, true);
  std::vector<float> x(cols), y0(rows), got(rows), deq(cols);
  for (auto &v : x)
    v = frnd();
  for (auto &v : y0)
    v = frnd();

  uint8_t *d_w;
  float *d_x, *d_y;
  ck(cudaMalloc(&d_w, w.size() + 16), "malloc w");
  ck(cudaMalloc(&d_x, (cols + 4) * 4), "malloc x");
  ck(cudaMalloc(&d_y, rows * 4), "malloc y");
  ck(cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice), "h2d w");
  float *xs = d_x + (unaligned_x ? 1 : 0); /* 4-byte aligned, not 16 */
  ck(cudaMemcpy(xs, x.data(), cols * 4, cudaMemcpyHostToDevice), "h2d x");
  ck(cudaMemcpy(d_y, y0.data(), rows * 4, cudaMemcpyHostToDevice), "h2d y");
  const int rc = sq::gemv(t, d_w, xs, d_y, rows, cols, accumulate, nullptr);
  ck(cudaGetLastError(), "gemv launch");
  ck(cudaMemcpy(got.data(), d_y, rows * 4, cudaMemcpyDeviceToHost), "d2h y");
  cudaFree(d_w);
  cudaFree(d_x);
  cudaFree(d_y);
  if (rc)
    return "gemv returned error";

  for (int r = 0; r < rows; r++) {
    if (spite_dequantize_row(t, w.data() + r * row_bytes, deq.data(), cols) !=
        0)
      return "reference dequant failed";
    double dot = 0, mag = 0;
    for (int c = 0; c < cols; c++) {
      const double p = static_cast<double>(deq[c]) * static_cast<double>(x[c]);
      dot += p;
      mag += fabs(p);
    }
    if (accumulate) {
      dot += y0[r];
      mag += fabs(y0[r]);
    }
    const double ae = fabs(static_cast<double>(got[r]) - dot);
    const double re = mag > 0 ? ae / mag : (ae == 0 ? 0 : INFINITY);
    if (!(ae <= INFINITY) || isnan(got[r]))
      return "non-finite gemv output";
    if (ae > err.max_abs)
      err.max_abs = ae;
    if (re > err.max_rel)
      err.max_rel = re;
  }
  return nullptr;
}

const char *check_rejects(SpiteType t) {
  const uint64_t sb = spite_type_block_bytes(t),
                 se = spite_type_block_elements(t);
  uint8_t *d_w;
  float *d_x, *d_y;
  ck(cudaMalloc(&d_w, 64 * 1024), "malloc");
  ck(cudaMalloc(&d_x, 64 * 1024), "malloc");
  ck(cudaMalloc(&d_y, 4096), "malloc");
  cudaMemset(d_w, 0, 64 * 1024);
  const char *why = nullptr;
  if (se > 1 && sq::gemv(t, d_w, d_x, d_y, 4, static_cast<int>(se) + 1, false,
                         nullptr) != -1)
    why = "cols not a multiple of the block size was accepted";
  else if (sq::gemv(t, d_w, d_x, d_y, 4, 0, false, nullptr) != -1)
    why = "cols == 0 accepted";
  else if (sq::gemv(t, d_w + 1, d_x, d_y, 4, static_cast<int>(se) * 2, false,
                    nullptr) !=
           (t == SPITE_TYPE_MXFP4 || t == SPITE_TYPE_NVFP4 ? 0 : -1))
    why = "odd-aligned weight pointer handled wrongly";
  (void)sb;
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaFree(d_w);
  cudaFree(d_x);
  cudaFree(d_y);
  return why;
}

// ── bandwidth ─────────────────────────────────────────────────────────────

double bench_gemv(SpiteType t, int rows, int cols) {
  const uint64_t sb = spite_type_block_bytes(t),
                 se = spite_type_block_elements(t);
  const size_t mat = sb * (cols / se) * static_cast<size_t>(rows);
  /* rotate through copies totalling >= 1 GiB so the 96 MB L2 never serves the
   * weights */
  const int ncopy = static_cast<int>((1ull << 30) / mat) + 2;
  std::vector<uint8_t *> w(ncopy);
  for (auto &p : w) {
    ck(cudaMalloc(&p, mat), "bench malloc w");
    ck(cudaMemset(p, 0x3C, mat), "bench memset");
  }
  float *d_x, *d_y;
  ck(cudaMalloc(&d_x, cols * 4), "malloc");
  ck(cudaMalloc(&d_y, rows * 4), "malloc");
  ck(cudaMemset(d_x, 0, cols * 4), "memset");
  for (int i = 0; i < 4; i++)
    sq::gemv(t, w[i % ncopy], d_x, d_y, rows, cols, false, nullptr);
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  const int iters = 100;
  cudaEventRecord(a);
  for (int i = 0; i < iters; i++)
    sq::gemv(t, w[i % ncopy], d_x, d_y, rows, cols, false, nullptr);
  cudaEventRecord(b);
  ck(cudaEventSynchronize(b), "bench sync");
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  for (auto p : w)
    cudaFree(p);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return static_cast<double>(mat) * iters / (ms * 1e-3) / 1e9;
}

} // namespace

int main(int argc, char **argv) {
  bool bench = false, bench_all = false;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--bench"))
      bench = true;
    else if (!strcmp(argv[i], "--bench-all"))
      bench = bench_all = true;
    else {
      fprintf(stderr, "usage: %s [--bench|--bench-all]\n", argv[0]);
      return 2;
    }
  }
  cudaDeviceProp prop;
  ck(cudaGetDeviceProperties(&prop, 0), "device props");
  printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

  int fails = 0, dec_ok = 0, gemv_ok = 0;
  double worst_abs = 0, worst_rel = 0;
  const char *worst_name = "";
  for (const TypeInfo &ti : ALL) {
    const char *e = nullptr;
    const uint64_t se = spite_type_block_elements(ti.t);
    if (spite_type_block_bytes(ti.t) != 0 && se != 0) {
      for (int pass = 0; pass < 2 && !e; pass++) {
        e = check_decode(ti.t, pass == 1);
        if (e && pass == 1) {
          static std::string m;
          m = std::string("(raw bytes) ") + e;
          e = m.c_str();
        }
      }
    } else
      e = "type has no block size";
    const bool dec_pass = e == nullptr;
    GemvErr err;
    if (!e)
      e = check_gemv(ti.t, 67, 4096, false, false, err);
    if (!e)
      e = check_gemv(ti.t, 67, 4096, true, false, err);
    if (!e)
      e = check_gemv(ti.t, 33, 4096, false, true, err);
    if (!e)
      e = check_gemv(ti.t, 9, se > 1 ? static_cast<int>(se) : 96, false, false,
                     err);
    if (!e)
      e = check_gemv(ti.t, 9, se > 1 ? 3 * static_cast<int>(se) : 97, true,
                     false, err);
    if (!e && err.max_rel > 1e-4)
      e = "gemv error above 1e-4";
    if (!e)
      e = check_rejects(ti.t);
    if (dec_pass)
      dec_ok++;
    if (!e)
      gemv_ok++;
    if (err.max_rel > worst_rel) {
      worst_rel = err.max_rel;
      worst_name = ti.name;
    }
    if (err.max_abs > worst_abs)
      worst_abs = err.max_abs;
    if (e) {
      fails++;
      printf("FAIL %-8s %s\n", ti.name, e);
    } else
      printf("PASS %-8s decode bit-exact (finite+raw)  gemv max_abs=%.3g "
             "max_rel=%.3g\n",
             ti.name, err.max_abs, err.max_rel);
  }
  if (sq::gemv(static_cast<SpiteType>(9999), nullptr, nullptr, nullptr, 1, 1,
               false, nullptr) != -1) {
    fails++;
    printf("FAIL unknown type not rejected\n");
  }
  printf("decode bit-exact: %d/%d types; gemv+rejects: %d/%d types; worst gemv "
         "rel err %.3g (%s), worst abs err %.3g\n",
         dec_ok, NTYPES, gemv_ok, NTYPES, worst_rel, worst_name, worst_abs);

  if (bench) {
    const int rows = 14336, cols = 4096;
    printf("\nGEMV bandwidth, %d x %d (rows x cols), weights rotate over >=1 "
           "GiB; peak ~1792 GB/s\n",
           rows, cols);
    for (const TypeInfo &ti : ALL) {
      const bool main5 = ti.t == SPITE_TYPE_Q4_K || ti.t == SPITE_TYPE_Q5_K ||
                         ti.t == SPITE_TYPE_Q6_K || ti.t == SPITE_TYPE_IQ4_XS ||
                         ti.t == SPITE_TYPE_Q8_0;
      if (!bench_all && !main5)
        continue;
      printf("  %-8s %7.1f GB/s\n", ti.name, bench_gemv(ti.t, rows, cols));
    }
  }
  printf("%s\n", fails ? "FAILED" : "PASSED");
  return fails != 0;
}
