# Porting PXA's Pascal work into spite (plan)

This tracks what we intend to take from [poisonxa16/pxa](https://github.com/poisonxa16/pxa)
and how, for the Tesla P100 (sm_60) path. pxa is a llama.cpp/ik_llama fork, **MIT**
licensed (its `NOTICE`/`LICENSING.md`); a few of its files are upstream llama.cpp
(MIT, © the ggml authors) rather than PXA-original. Anything we lift keeps its
copyright header and is listed in [`core/THIRD_PARTY.md`](../core/THIRD_PARTY.md).

## Why we can't just copy files

pxa is a ggml-graph engine. Its biggest P100 wins are **graph-fusion** (collapsing
runs of glue kernels) and the **PXQ/PXQN codec**, neither of which maps onto
spite's ABI-op kernel model. Concretely, two of pxa's headline P100 levers are
already redundant here:

| pxa lever | why it doesn't transfer |
|---|---|
| `PXA_FUSE_DELTANET` (DeltaNet glue fusion, +3.7% P100 decode) | spite's `linear_attn` already fuses q/k L2-norm, gate/beta, the delta update, gated RMS norm and the silu gate into `gdn_core_kernel` |
| `pxa-ew-fuse` (elementwise launch overhead) | spite is already one fused kernel per op; it has no long graph of trivial kernels |

Also note the fastest P100 kernels are **not** in the public tree: `PXA_PXQ4_RB`
and the PXQN tiers ship only in pxa's closed release library (`docs/LEVERS.md`).
A from-source pxa build is ~12% slower on PXQ4 decode than its shipped build.

## What is worth porting

### 1. MTP speculative decoding (`spite run --mtp`) — **blocked on batching**

pxa runs MTP as a batched verify: draft K tokens with the NextN head, verify all
K+1 in one target pass. Its `PXA_REP_GUARD_LAZY` note is the useful part —
a repetition guard that penalises the draft target drops MTP acceptance from 64%
to 27% (fn32, one P100), which is why greedy MTP must verify the *raw* argmax.

**Blocker:** `HybridDecoder::forward` runs one token at a time
(`crates/spite-models/src/hybrid.rs`), so a "verify K+1 tokens" call still costs
K+1 sequential passes — there is no amortisation, so MTP would be strictly
slower. Worse, the 48 Gated-Delta-Net layers carry a *sequential recurrent state*
that a batched verify would advance past the accepted prefix; correct MTP needs a
batched GDN forward plus state snapshot/rollback. That is the same batched-forward
work as item 2.

### 2. Fast prefill — **blocked on batching (same work as item 1)**

pxa's P100 finding (`src/llama-build-context.cpp`, `docs/LAUNCHER.md`): on
pre-Turing cards the **non-flash** batched-cuBLAS attention chain is the fast
*prefill* regime (P100 fa-off **1213** vs fa-on **817 t/s**), while flash
attention is the fast *decode* regime. spite's prefill is decode-speed
(6.86 t/s on the tester's box) because `forward` is token-at-a-time, not because
of a kernel choice. Fixing it means a batched prefill forward: batched projections
(GEMM), a causal prefill attention, and a batched (chunked-scan) GDN recurrence.
Large, and needs P100 hardware to tune and bench.

### 3. sm_60 attention for the wide-GQA shape (portable, needs P100 to verify)

pxa's `docs/DELTA-SINCE-FORK-POINT.md` lists P100-specific attention work that
matches Qwen3.8 exactly (head_dim 256, 24 q heads / 4 kv heads):

* `PXA_FA_GQA_PACK` — head-packed D=256 decode kernel for sm_60, with a 4-way ILP
  V-pass and an optional shared-memory staging variant (`PXA_FA_GQA_QSMEM`).
* `PXA_FA_D512_CHAIN_F32` — accumulate wide-head attention in fp32 (accuracy).
* mask-skip tile variants (`PXA_FA_MASK_SKIP_TILE*`).

These map onto `kernels/qwen/qwen3_5/nvidia/sm_60/tesla_p100/` (or the vendor
`nvidia/attn.cu` if arch-wide). They are CUDA kernels: they must be built with a
CUDA 12.x toolkit and pass `tools/verify/verify.py` plus a `spite-bench` before/after
on the P100s. They cannot be compiled or verified on a CUDA 13 host (sm_60 dropped).

### 4. `pxa-fastdiv` (small, GP100 has no 64-bit integer divider)

Header-only multiply+shift division by a loop-invariant divisor. Worth taking only
where a hot 64-bit division in index math is found by profiling; do not add it
speculatively.

### 5. PXQ-style low-bit codec (largest)

The only way to fit a 27B model on one 16 GB P100 (Q6_K is ~20.9 GiB). A port means
new `SpiteType` ids, a CPU codec, CUDA GEMV/GEMM, GGUF type mapping and the
quantizer — ABI-additive but cross-cutting. PXQ4 is MXFP4-compatible (32-element
blocks, 4.25 bits/weight), so the format itself is simple; the work is the surface
area. Start on the CPU codec (testable without a GPU).

## Building and verifying sm_60 without a Pascal card

The host toolkit here is CUDA 13.x, which dropped sm_60, but a CUDA 12.x
container builds it and the host GPU can still *run* the result through PTX
JIT — enough to gate **correctness** on kernel changes without a P100. (Speed
still needs the P100s; JIT on a different arch is not a P100 measurement.)

```bash
# CUDA 12.6 has nvcc for sm_60; the host only has CUDA 13.
docker run -d --name spite-p100 -v "$PWD":/work -w /work \
  nvidia/cuda:12.6.3-devel-ubuntu22.04 sleep infinity
docker exec spite-p100 bash -lc \
  'apt-get update -qq && apt-get install -y -qq cmake python3 python3-pip && pip install -q cmake'

# Build the P100 kernels; also emit PTX so the host driver can JIT them.
docker exec spite-p100 bash -lc 'rm -rf /tmp/build && \
  cmake -S /work -B /tmp/build -DSPITE_MODELS=qwen/qwen3_5 \
    -DSPITE_GPU_ARCHS=TESLA_P100 -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_FLAGS="--generate-code=arch=compute_60,code=compute_60" && \
  cmake --build /tmp/build -j$(nproc)'

# Pull the .so and a CUDA 12 libcudart out, then verify on the host GPU.
docker cp spite-p100:/tmp/build/kernels /tmp/p100-build/kernels
docker cp spite-p100:/usr/local/cuda/lib64/libcudart.so.12.6.77 /tmp/p100-build/lib/
ln -sf libcudart.so.12.6.77 /tmp/p100-build/lib/libcudart.so
LD_LIBRARY_PATH=/tmp/p100-build/lib python3 tools/verify/verify.py \
  /tmp/p100-build/kernels/qwen/qwen3_5/nvidia/libkernel_qwen_qwen3_5_nvidia.so
```

`cuobjdump --list-elf` shows the `sm_60` cubins; `--list-ptx` shows the PTX the
host JITs. `verify.py` passed on the current vendor kernel this way (ABI v7,
93 OK / 12 SKIP / 0 FAIL on the tester's box).

## Status

- **Peer-capable tensor split: done.** pxa measures `-sm tensor` on a 2×
  P100-PCIE **PHB** pair at +17–27% decode / +44% prefill over the layer split, so
  `p100_multi` now gates tensor parallelism on *peer access*
  (`cudaDeviceCanAccessPeer`), not NVLink. NVLink vs PCIe changes the speed, not
  whether the split is allowed. (`crates/spite-parallel/src/p100_multi.rs`.)
- **Batched hybrid prefill: done (needs P100 before/after).** The engine and the
  sm_60 kernels both implement it:
  - generic reference ops accept `[cols, m]` (`5430ec4`), and
    `HybridDecoder::forward` runs a multi-token prompt layer-major via
    `forward_batch` (`a9bb561`).
  - the CUDA vendor ops batch too: `q35_gemv_batch` (multi-column GEMV) drives
    `ffn`, `matmul`, the `attention_ex` projections and the `linear_attn`
    projections (`a095efe`, `1e4bbff`, `3a61c99`); attention and the GDN
    recurrence stay per token, so the weight stream is read once per 4-column
    chunk instead of once per token.
  - an optional `spite_kernel_caps()` bit (`SPITE_CAP_BATCH`, no ABI bump) lets a
    kernel advertise batch support; the dispatcher reads it and
    `batch_capable()` gates on it (`290c102`).
  - a **pipeline split batches too**: `forward_batch` runs one op per layer with
    a per-stage `[d, m]` activation and hands the whole block between stages
    through the same host hop as the 1-token path, and `batch_capable()` no
    longer requires every stage on one device. Without this a 27B that needs
    two 16 GB P100s (always split) never took the batched path.
  - verified with `tools/verify/verify_batch.py` (CPU) and
    `tools/verify/verify_batch_cuda.py` (device, PTX JIT): ffn/matmul ~8e-8 vs a
    float64 oracle, attention_ex/linear_attn bit-identical to sequential m=1 with
    KV/state carried. `verify.py` (m=1) still passes; Qwen3.8-27B runs end to end
    on the host GPU through the sm_60 build.
  - **measured on the 2× P100-PCIE (PHB) box, 512-token prompt: 7.24 → 7.89
    prefill tok/s (`f1cc494` → `f85a2a2`, TTFT 70.7 → 64.9 s, 1.09×).** The
    kernel `.so` is byte-identical between the two heads, so this is the host
    path alone. It also says the GEMV's ~4× weight-traffic cut is *not* what
    bounds prefill: 9% of the time is all it bought, so per-token/per-op cost
    dominates (profiling plan in the PR). Prefill is still ~15× off
    llama.cpp's 117.5 tok/s on the same box and file.
  - **still needs the P100s**: a `spite-bench` before/after posted in the PR
    description — the `.bench` carries the measured rows, but the
    generic-fallback "before" row is still missing.
  - **MTP prompt prefill**: a prefill now fills the draft block's own KV cache
    over the prompt (one draft pass per prompt token, ~1–2% of prefill) instead
    of leaving those rows unwritten; see `HybridDecoder::forward`.
  - **MTP prompt prefill is batched**: the per-token draft pass it used to run
    (a measured −6.6% on the 512-token prefill column) is gone. The stem, the
    NextN block and `matmul` were already `SPITE_CAP_BATCH`, so
    `HybridDecoder::mtp_prefill_batch` runs the whole prompt through the NextN
    block in one causal pass and skips the shared head (its logits are
    discarded). `prefill_fills_the_mtp_kv` pins it to the per-token path.
- **Batched MTP speculative verify: done.** The old loop ran one trunk
  `decode_step` per accepted draft, so MTP cost a full trunk pass per emitted
  token *plus* the NextN head — slower than plain decode at any K. The verify
  is now one batched trunk pass over `[tok, d0..dK-1]` (m = K+1) through
  `ModelArch::verify_batch`, followed by `rollback_drafts`.
  - The GDN conv/delta state has no positional index, so it is snapshotted
    before the pass (`DeviceBuffer::copy_from`, a device-to-device copy) and
    restored when a draft tail is rejected; the KV cache needs no snapshot
    (rows are position-indexed and overwritten). `rollback_drafts` also
    reselects the accepted column's final-normed hidden, which the next
    `mtp_step` consumes; `forward_batch` stashes all columns for that.
  - `rollback_undoes_the_rejected_tail` (CPU) runs a verify whose tail is then
    dropped and checks the following token against a prefix-only reference.
  - Still owed: the P100 before/after. The verify is one pass instead of K+1,
    so it should finally beat plain decode, but that is unmeasured here.
- **Batched GEMV: chunk widened 4 → 8 (measured), the shared-memory rewrite
  rejected.** `gemv_batch_kernel` decodes a weight row `m/kBatchChunk` times, so
  `kBatchChunk` 4 → 8 halves the redundant dequant with the *same* per-column
  accumulation order (bit-identical to the m=1 row kernel; `verify_batch_cuda.py`
  stays at 0.0e+00). Measured on the 27B FFN-gate shape (rows 17408, cols 5120,
  Q6_K): **19.45 → 16.00 ms**, 1.22×. 16 regressed (register pressure), so 8 is
  the default; `SPITE_GEMV_BATCH_CHUNK` overrides it for A/B.
  - The reviewer's suggested fix — dequantize once per tile into shared memory
    and reuse across columns — was implemented and **measured 16× slower**
    (304 vs 19.4 ms at m=512). Each thread read its column's activations with a
    `cols` stride between columns, i.e. uncoalesced; the FMA loop was starved on
    activation traffic rather than helped by the removed decode. A reusable
    version needs activation staging too (a real tiled GEMM with both operands
    in shared memory, register-blocked accumulators), which is arch-specific and
    must be tuned on the P100. The negative result is kept in the log: the
    dequant is not the only cost, and per-thread column ownership is the trap.
  - Both measurements are on the host RTX 5090 through PTX JIT of the sm_60
    build, not the P100s; re-measure there before trusting the factor.
- **The activation stream is the real wall, and row-tiling is what moves it.**
  The chunk above was tuning the wrong operand. A probe that replaces the
  activation loads with a constant takes the FFN-gate shape from **16.7 → 4.1 ms
  (75% of the kernel)**, while sweeping `kBatchChunk` 2/4/8/16 over the same
  shape gives 18.2/19.8/16.8/19.3 ms — U-shaped, i.e. a register-pressure
  tradeoff, not a bandwidth one. The reason is structural: every one of the
  `rows` blocks reads the **whole** `[cols, m]` activation, so those bytes are
  `rows`-fold amplified — 182.5 GB against 4.68 GB of weights (39×) at this
  shape. Each activation float4 a thread loads feeds exactly one FMA, so there
  is no reuse to recover; staging the activation in shared memory does not help,
  because it changes *where* the load comes from, not *how often* the value is
  used. (That is the rejected smem rewrite above seen from the other end, and it
  is what `Kmic-68/llama.cpp`'s `OPTLOG.md` reports for `mul_mat_vec_q` on
  sm_60: attempts 43 and 47-51, every block-wide or reduced-activation-traffic
  variant, lost on occupancy — "what binds is warps resident per SM" — while
  `split_rows`, where rows *share* one activation image, won.)
  - Fix: `kRowTile` output rows per block, the activation kept in registers and
    reused across them. Two instantiations are launched: `(kRowTile 4,
    kRowWideChunk 4)` when `rows/4 >= 256` blocks, `(1, kBatchChunk 8)`
    otherwise. The chunk has to follow the tile, because the two forms want
    opposite answers (sm_60 ptxas: 79 registers at (1, 8) = 3 blocks/SM, 162 at
    (4, 8) = **1** block/SM, 128 at (4, 4) = 2 blocks/SM; register pressure is
    not monotone in the tile, so these have to be read off `ptxas -arch=sm_60
    -v`). Per-column accumulation order is untouched, so the result is still
    bit-identical to the m=1 row kernel — `verify_batch_cuda.py` stays at
    0.0e+00, and a sha256 of `y` over five shapes (surplus rows and
    `accumulate` included, tiled form forced) matches the unmodified kernel.
  - Measured on the host RTX 5090 through PTX JIT of this sm_60 build (harness
    in `/tmp`, cols 5120, m 512 and the verify width m 5):

    | projection | rows | m=512 before → after | m=5 before → after |
    |---|---|---|---|
    | FFN gate/up | 17408 | 15.94 → 10.09 ms (**1.58×**) | 0.164 → 0.133 ms (1.23×) |
    | FFN down | 5120 | 12.29 → 8.65 ms (1.42×) | 0.125 → 0.102 ms (1.23×) |
    | attention q | 12288 | 10.89 → 7.42 ms (1.47×) | 0.117 → 0.098 ms (1.19×) |
    | attention o | 6144 | 5.63 → 3.85 ms (1.46×) | 0.063 → 0.053 ms (1.19×) |
    | attention k/v | 1024 | 1.18 → 0.83 ms (1.41×) | 0.017 → 0.014 ms (1.21×) |
    | GDN qkv | 2560 | 2.38 → 1.59 ms (1.50×) | 0.028 → 0.024 ms (1.17×) |
    | GDN gate/out | 2048 | 2.24 → 1.57 ms (1.43×) | 0.027 → 0.024 ms (1.12×) |
    | GDN beta/alpha | 32 | 0.539 → 0.426 ms (1.27×) | 0.008 ms (1.00×) |

    A `rows` sweep 16 → 17408 found no band where the change loses (worst 1.04×
    at 640 rows, and 1.43×-1.64× from 1024 up). The 256-block threshold is
    deliberately on the safe side of a crossover measured between 160 and 192.
    `verify.py`: 93 OK / 12 SKIP / 0 FAIL, same as before.
  - Still owed: the P100 before/after. The threshold and the chunk/tile pair are
    *host* measurements — 170 SMs and 96 MB of L2 against the P100's 56 and
    4 MB, so the crossover can move. Also unmeasured here: the same `rows`-fold
    amplification in the m=1 `gemv_row_kernel` (356 MB of activation against
    73 MB of weights at this shape), which is the decode path and takes the same
    fix.
- **MTP hidden-state input confirmed (reviewer lead closed).** The suspicion was
  that `mtp_step` double-normalizes by feeding the `output_norm`'d hidden into
  `hnorm`. llama.cpp's `src/models/qwen35.cpp` stores exactly that
  (`t_h_nextn = build_norm(cur, model.output_norm)`) and its MTP graph applies
  `hnorm` to it, so the current input is correct; the pre-final-norm residual
  would be wrong. The remaining question — why acceptance is ~100% against
  llama.cpp's 84% — is not the hidden state, so a discriminator was added:
  `SpecStats::draft_trunk_tv_sum` (mean total-variation distance between the
  draft and trunk distributions over the first few verified positions). Near
  zero means the head reproduces the trunk's own prediction (a high acceptance
  rate cannot tell that apart from a good head); `spite-bench --mtp` prints it
  as `draft-vs-trunk TV`. On the tiny fixture it is 0.11, i.e. the head does
  real work there; the 27B number will say whether the real head does.
- **CPU "before" gate row: still missing, and its cause is upstream of the
  P100 box.** `DenseWeights::load` dequantizes every tensor to F32, so a 27B
  needs ~108 GB of RAM just to *load* on the CPU and the generic-fallback row
  cannot be produced on the tester's 15 GiB box. Options, cheapest first: emit
  the row on a smaller qwen35 model (different model, must be labelled), or
  give the CPU reference an on-demand dequant path that keeps packed bytes and
  expands per matvec (a real memory win for every CPU user, but it touches
  `Weight` and every arch impl). Until one lands the PR cannot satisfy the
  AGENTS.md gate on a machine like the tester's.
- **PXQ4 = MXFP4: done.** PXQ4 is pxa's repack of MXFP4 (ggml type 39): 32-element
  blocks, E8M0 scale, e2m1 codes, 4.25 bpw — the tier that fits a 27B on one
  16 GB P100. Inference already worked in spite (CPU `dq_mxfp4`, the CUDA
  `MXFP4` decoder, the loader); this session added the missing **quantizer** and
  a **GGUF v3 writer** (`crates/spite-quantize/src/{mxfp4,gguf_write}.rs`), so
  `spite-quantize --type MXFP4` produces a file. Verified: codec round-trip,
  the tiny fixture quantized/reloaded/decoded on CPU, and MXFP4 matmul exercised
  by `verify.py` on the sm_60 kernel. 1-D parameters (GDN dt/a/norm, conv and
  RMS norm weights) stay F32, as the ops require.
- **sm_60 GEMV tuning**: started on the batched GEMV — the activation stream is
  the wall and row-tiling moves it (the entry above carries the numbers). What is
  left is measurement-driven and needs the P100s: the 256-block threshold, the
  `(kRowTile 4, kRowWideChunk 4)` pair, and the same tiling for the m=1
  `gemv_row_kernel`. sm_60 attention tuning is not started.
