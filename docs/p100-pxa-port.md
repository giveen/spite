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

## Status

- **Item (peer-capable tensor split): done.** pxa measures `-sm tensor` on a 2×
  P100-PCIE **PHB** pair at +17–27% decode / +44% prefill over the layer split, so
  `p100_multi` now gates tensor parallelism on *peer access*
  (`cudaDeviceCanAccessPeer`), not NVLink. NVLink vs PCIe changes the speed, not
  whether the split is allowed. (`crates/spite-parallel/src/p100_multi.rs`.)
- Items 1 and 2 share the batched-hybrid-forward dependency; 3–5 are kernel/codec
  work that needs the P100s to build, verify and benchmark.
