# Contributing a Kernel

You have a GPU. You want it to run inference faster. This is the guide.

## The idea

spite ships a fallback kernel that runs on any CUDA GPU. It's correct but
unoptimized. If you write a better kernel for your specific card, everybody
with that card benefits. You don't need to understand the whole codebase —
just your GPU and one operation.

## Step 1 — Find out what's slow

```bash
spite dispatch -m path/to/model.gguf --card RTX_3090
```

Output looks like:

```
  rms_norm       → sm_89/kernels/llama/llama4/sm_89
  attention      → generic/kernels/generic/generic
  ffn            → generic/kernels/generic/generic
  layer          → generic/kernels/generic/generic
  spec_verify    → generic/kernels/generic/generic
  prefill        → generic/kernels/generic/generic
```

Entries resolved to `generic/` are the opportunities — a kernel tuned for
your card would be faster.

## Step 2 — Copy the template

```bash
cp kernels/llama/llama4/nvidia/sm_89/rtx_4090/KERNEL_TEMPLATE.cu \
   kernels/llama/llama4/nvidia/sm_120/rtx_5090/attention.cu
```

Edit the `gpu_arch` field in `kernel_info` at the bottom of the file.

The template has comments explaining the block layout, tensor core shapes,
and what each op does. Read them. The Q4_K block layout comment especially —
getting dequantization wrong is the most common mistake.

## Step 3 — Implement one op

You don't need to implement everything. Set the ops you haven't written to
`NULL` in the `kernel_info` struct — the dispatcher uses the fallback for
those and your kernel for the ones you did implement.

Start with `ffn` or `rms_norm`. Attention is the most complex.

## Step 4 — Build and verify correctness

```bash
cmake -B build -DSPITE_MODELS="llama/llama4" -DSPITE_GPU_ARCHS="RTX_5090" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build

python3 tools/verify/verify.py \
  build/kernels/llama/llama4/nvidia/sm_120/rtx_5090/libkernel_llama_llama4_nvidia_sm_120_rtx_5090.so
```

This runs your kernel against the generic reference implementation on a set
of random inputs and checks that outputs match within tolerance. It must pass
before a PR will be accepted.

## Step 5 — Benchmark

```bash
cargo run --release -p spite-bench -- --model path/to/model.gguf
```

**Copy this output into your PR description.** It's the record of what your
kernel achieves on your hardware.

## Step 6 — Submit the PR

```
kernels/
  llama/
    llama4/
      nvidia/
        sm_120/
          rtx_5090/
            attention.cu      ← your file
            attention.bench   ← the bench output
```

PR title format: `kernel: llama/llama4/nvidia/sm_120/rtx_5090 attention`

That's it. No need to touch anything outside the `kernels/` directory.

---

## FAQ

**What GPU architectures are supported?**

See `docs/gpu_guides/` for a guide per architecture with the tensor core
shapes, shared memory limits, and known-good tile sizes.

**My kernel is faster on batch=1 but slower on batch=8. Should I submit it?**

Yes — note it in the PR. The dispatcher can eventually select kernels based
on batch size. Right now it picks one, so mention which use case you optimized
for.

**Can I contribute for AMD / Intel / Apple Silicon?**

Yes. AMD kernels go in `rdna2/`, `rdna3/` and use HIP (mostly a find/replace
of `cuda` → `hip`). Intel Arc goes in `arc_alchemist/` and uses SYCL or
OpenCL. Metal kernels for Apple Silicon go in `metal/` and use `.metal` files.
See the relevant guide in `docs/gpu_guides/`.

**Do I need to handle every quant type?**

No. Declare only the types your kernel handles in `supported_quants`. Others
fall back automatically.

**The verify tool says my outputs don't match. What tolerance is acceptable?**

For Q4_K: max absolute error < 0.05, mean absolute error < 0.01. FP16: max
< 1e-3. The verify tool prints the actual errors so you can see how close you
are.
