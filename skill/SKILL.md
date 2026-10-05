---
name: spite-building-guide
description: Building, verifying, benchmarking, and review guide for spite kernels and host engine, following the Linus Torvalds review methodology.
version: 1.0.0
---

# Spite Building & Review Guide

This skill defines the technical procedures for building, testing, verifying, and benchmarking spite, coupled with a strict review methodology adapted from Linus Torvalds' principles. It ensures the spite engine remains modular, correct, fast, and free of regressions.

---

## 1. Reviewer & Contributor Mindset

1. **"My job is to say no."**
   - *Principle*: Rejecting unverified, slow, or architectural-violating code protects the engine. Merging convenience hacks degrades performance and breaks maintainability.
2. **"Good programmers worry about data structures, not code."**
   - *Principle*: Tensor layouts, stride alignment, cache lines, shared memory bank conflicts, and WMMA/matrix-multiply fragment shapes dictate GPU efficiency. If the memory access pattern is bad, clever code will not save it.
3. **"Talk is cheap. Show me the code — and the benchmark."**
   - *Principle*: Latency claims are meaningless without reproducible `.bench` files comparing against the generic fallback. Correctness claims are meaningless without `verify.py` passing.
4. **"Prefer correctness over cleverness."**
   - *Principle*: A correct, simple reference kernel beats an unstable kernel that silently outputs garbage logits or crashes under edge-case context lengths. Correct first, fast later.
5. **"Never break the ABI silently."**
   - *Principle*: `crates/spite-abi` and `core/abi.h` form the immutable contract between the host and dynamically loaded `.so` kernels. Any structural change requires an explicit `ABI_VERSION` bump.

---

## 2. Review Triggers & Non-Negotiables

### Level 1 — Global Invariants (Immediate Rejection)

Any violation of these triggers must be **rejected**:

- **Trigger: Unverified Kernel Output**
  - *What to look for*: PR opened without running `tools/verify/verify.py` or with output exceeding numerical tolerance (`1e-4` for f16/f32, `1e-3` for Q8).
  - *Why*: A kernel that produces incorrect tokens faster is a regression, not an optimization.
  - *Severity*: **REJECT**

- **Trigger: Missing or Falsified Benchmark Gate**
  - *What to look for*: Missing `.bench` file in the card directory or PR description omitting the before/after comparison table against `kernels/generic/generic/`.
  - *Why*: Improvements must be measurable on real hardware. Slower kernels are regressions.
  - *Severity*: **REJECT**

- **Trigger: Breaking ABI Without Bumping `ABI_VERSION`**
  - *What to look for*: Changes to `SpiteTensor`, `SpiteCtx`, `SpiteKernelInfo`, function pointer signatures, or enum values in `crates/spite-abi/src/lib.rs` or `core/abi.h` without incrementing `ABI_VERSION`.
  - *Why*: Existing pre-compiled kernels will segfault or misinterpret memory layouts upon `dlopen()`.
  - *Severity*: **REJECT**

- **Trigger: Pushing Directly to `main` or `master`**
  - *What to look for*: Direct commits pushed to primary branches without branch PR workflow.
  - *Why*: Bypasses CI, peer review, and verification gates.
  - *Severity*: **REJECT**

- **Trigger: Skipping or Disabling Tests to Pass CI**
  - *What to look for*: `#[ignore]` added to failing tests, deleting assert checks, or lowering error thresholds.
  - *Why*: Masking failure hides defects from developers.
  - *Severity*: **REJECT**

- **Trigger: Fatal Abort / Panic on Recoverable Data**
  - *What to look for*: `panic!`, `unwrap()`, or `assert!` on user-supplied GGUF tensors, token IDs, or socket inputs.
  - *Why*: Server should return descriptive errors or fall back to generic ops, never crash the host process.
  - *Severity*: **REJECT**

- **Trigger: Leaking Device Resources or Missing Synchronization**
  - *What to look for*: Allocating device memory or scratchpads without freeing, ignoring CUDA streams in `SpiteCtx`, or missing stream synchronizations on host-device transfers.
  - *Why*: Causes out-of-memory crashes and data races across concurrent inference requests.
  - *Severity*: **REJECT**

---

### Level 2 — Structural & Architectural Patterns (Request Changes)

- **Trigger: Misplaced Kernel Scope**
  - *Rule*: Code must live at the exact scope of its benefit, resolved by answering 4 guiding questions:
    1. *Does this benefit EVERYONE running that model on any hardware?* → `kernels/generic/generic/` (or `kernels/<family>/<model>/`).
    2. *Does this benefit EVERYONE running that BRAND of card (NVIDIA, AMD, Intel, Apple) for this model?* → `kernels/<family>/<model>/<company>/` (vendor root of that model tree, e.g. `qwen/qwen3/nvidia/`).
    3. *Does this benefit EVERYONE running that specific GPU ARCHITECTURE (`sm_89`, `sm_120`, `rdna3`)?* → `kernels/<family>/<model>/<company>/<arch>/` (or `kernels/generic/<company>/<arch>/` if cross-model).
    4. *Does this benefit ONLY people running that specific CARD (`rtx_5090`, `rtx_3060`, `rx_7900_xtx`)?* → `kernels/<family>/<model>/<company>/<arch>/<card>/` (narrowest sub-path).
  - *Why*: Prevents architecture directories from being polluted with card-specific tile constraints and ensures vendor-wide baselines are placed at the tree root.
  - *Severity*: **REQUEST CHANGES**

- **Trigger: Unpinned or Split Rust Dependencies**
  - *What to look for*: Adding dependencies directly in a crate `Cargo.toml` without specifying `{ workspace = true }` and updating root `[workspace.dependencies]`.
  - *Why*: Divergent dependency versions bloat binary size and cause trait incompatibility.
  - *Severity*: **REQUEST CHANGES**

- **Trigger: Host Coupling to Specific Kernel Implementations**
  - *What to look for*: Adding hardcoded references to specific card `.so` files in `spite-dispatch`, `spite-models`, or `spite-server`.
  - *Why*: The dispatcher must discover kernels purely from the directory structure and metadata.
  - *Severity*: **REQUEST CHANGES**

---

### Level 3 — Code Hygiene & Cleanliness

- **Trigger: Lint or Formatting Failures**
  - *Check*: `cargo fmt --check` and `cargo clippy --workspace --all-targets -- -D warnings`.
  - *Severity*: **REQUEST CHANGES**

- **Trigger: Undocumented Card-Specific Constants**
  - *Check*: Magic numbers for shared memory sizes, block dimensions, or loop unrolls without comments explaining hardware limits (e.g. L2 cache size, register pressure).
  - *Severity*: **REQUEST CHANGES**

---

## 3. Building Spite

### Prerequisites

- **Host**: Linux x86_64 or aarch64
- **Rust**: 1.82+ (or latest stable)
- **C/C++**: C++23 compiler (`clang++` or `g++-13+`)
- **CMake**: 3.25+
- **GPU Toolkits**:
  - NVIDIA: CUDA 12.3+ (nvcc)
  - AMD: ROCm 6.0+ (hipcc)

---

### Building the Rust Host Workspace

```bash
# Build all crates in debug
cargo build

# Build all crates optimized for release
cargo build --release

# Run all unit, integration, and doc tests
cargo test --workspace

# Lint with strict warnings
cargo clippy --workspace --all-targets -- -D warnings

# Format check
cargo fmt --check
```

---

### Building Kernels with CMake

Kernels are compiled into standalone `.so` libraries and placed in their respective card directories.

```bash
# Configure for specific model and GPU card
cmake -B build \
  -DSPITE_MODELS="qwen/qwen3" \
  -DSPITE_GPU_ARCHS="RTX_5090" \
  -DCMAKE_BUILD_TYPE=Release

# Build kernels in parallel
cmake --build build -j$(nproc)

# Install into repository tree so spite-dispatch discovers them
cmake --install build --prefix .
```

Output binary:
`build/kernels/<family>/<model>/<company>/<arch>/<card>/libkernel_<family>_<model>_<company>_<arch>_<card>.so`

Installed binary:
`kernels/<family>/<model>/<company>/<arch>/<card>/libkernel_<family>_<model>_<company>_<arch>_<card>.so`

---

## 4. Verification Protocol (Correctness Gate)

Every compiled kernel must pass verification before opening a PR:

```bash
python3 tools/verify/verify.py \
  build/kernels/<family>/<model>/<company>/<arch>/<card>/libkernel_<family>_<model>_<company>_<arch>_<card>.so
```

### What `verify.py` Validates
1. **ABI Check**: Matches `SPITE_ABI_VERSION` (currently `4`).
2. **Op Correctness**: Tests `rms_norm`, `matmul`, `ffn`, and `attention` with pseudo-random tensors against the generic reference implementation.
3. **Tolerance**:
   - `max_abs_diff <= 1e-4` for FP16 and FP32 outputs.
   - `max_abs_diff <= 1e-3` for Q8_0 weights.
4. **Clean Exit**: Exits with code `0`.

---

## 5. Benchmarking Protocol (Performance Gate)

Benchmark the kernel against real model weights:

```bash
# Using the bench wrapper
python3 tools/benchmark/bench.py \
  --model /path/to/model.gguf \
  --n-tokens 16 \
  --device cuda \
  --card RTX_5090

# Or directly generating the .bench file
cargo run --release -p spite-bench -- \
  --model /path/to/model.gguf \
  --n-tokens 16 \
  --device cuda \
  > kernels/<family>/<model>/<company>/<arch>/<card>/<model_name>.bench
```

### Required PR Description Format

```markdown
## GPU
NVIDIA GeForce RTX 5090 (Driver: 570.86.16, CUDA 13.3)

## Operation
attention / ffn / rms_norm / full decode

## Technique
Shared memory tiling with vectorized half2 loads and warp-level reductions.

## Benchmark (before vs after)
| operation | kernel | latency (µs) / tok/s |
|---|---|---|
| decode | generic/generic (before) | 1.39 tok/s |
| decode | qwen/qwen3/nvidia/sm_120/rtx_5090 (after) | 95.7 tok/s |

## Verify Output
PASSED — all checks OK
```
