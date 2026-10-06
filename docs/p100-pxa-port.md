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
    recurrence stay per token, the weight stream is amortised over m.
  - an optional `spite_kernel_caps()` bit (`SPITE_CAP_BATCH`, no ABI bump) lets a
    kernel advertise batch support; the dispatcher reads it and
    `batch_capable()` gates on it (`290c102`).
  - verified with `tools/verify/verify_batch.py` (CPU) and
    `tools/verify/verify_batch_cuda.py` (device, PTX JIT): ffn/matmul ~8e-8 vs a
    float64 oracle, attention_ex/linear_attn bit-identical to sequential m=1 with
    KV/state carried. `verify.py` (m=1) still passes; Qwen3.8-27B runs end to end
    on the host GPU through the sm_60 build.
  - **still needs the P100s**: a `spite-bench` prefill before/after (the win is
    expected but unmeasured) and a `.bench` refresh.
- **PXQ codec, sm_60 attention/GEMV tuning**: not started (see the sections above).
