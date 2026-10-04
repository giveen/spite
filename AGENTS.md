# Agent Instructions for spite

Agents may open pull requests, push commits, and run `spite verify` and `spite bench`.
Use the GitHub MCP tools. Do **not** push to `main` or `master` directly.

---

## Architecture in one paragraph

spite is a Rust workspace where every layer is replaceable without touching any other.
The Rust host (`spite-loader` → `spite-dispatch` → `spite-executor` → `spite-scheduler` → `spite-server`)
loads `.gguf` weights and dispatches each operation (attention, FFN, rms\_norm) to the best
available kernel `.so` at runtime. Kernels live entirely in `kernels/<model>/<gpu_arch>/`.
Adding a new kernel file never changes any other kernel or any Rust code — the dispatcher
finds it automatically. Kernel development requires knowing one GPU; nothing else.

---

## Where does new code belong?

Before writing any code, answer this question about the improvement:

| Who benefits? | Where the code lives |
|---|---|
| Every model on every GPU (algorithmic improvement, correctness fix) | `kernels/generic/generic/` |
| One model family on every GPU (architecture-specific math) | `kernels/<model>/generic/` |
| One GPU architecture across all models | `kernels/<model>/<arch>/` |
| One specific card variant (tile sizes, cache layout, ISA quirk) | `kernels/<model>/<arch>/` (narrowest sub-path) |
| Rust host / ABI / scheduling / sampling | `crates/spite-<name>/src/` |

**Place code at the scope of its benefit.** A tile-size tweak that only helps the RTX 3060 does not belong in the sm\_86 directory — it belongs in a variant file with a comment explaining the card constraint. A math fix that applies to all LLaMA-family models does not belong in `sm_89/`; it belongs in `llama3/generic/` or `generic/generic/` so every GPU gets the improvement automatically.

---

## Change types and required checks

| Change type | Required before merge |
|---|---|
| New kernel (any GPU, any model) | `spite verify <path>` passes, `.bench` file included, before/after numbers in PR |
| Kernel modification | Same as new kernel |
| Rust crate change | `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test --workspace` |
| ABI change (`spite-abi/src/lib.rs`) | Bump `ABI_VERSION`, note in PR description which kernels must be recompiled |
| Documentation only | No code checks required |

---

## Benchmark gate (required for all kernel PRs)

Every kernel PR must include a `.bench` file alongside the kernel source.
Generate it with:

```bash
spite bench kernels/<model>/<arch>/<op>.cu > kernels/<model>/<arch>/<op>.bench
```

The PR description must contain a before/after table. The "before" row is always
the generic fallback from `kernels/generic/generic/`. Example format:

```
| operation | kernel | latency (µs) |
|---|---|---|
| attention | generic/generic (before) | 841.2 |
| attention | llama3/sm_86 (after)     | 213.7 |
```

**A PR without before/after numbers will not be reviewed.**
The improvement must be present and measurable. A new kernel that is slower than the
generic fallback is a regression, not a contribution.

---

## Kernel correctness (required before benchmarking)

```bash
spite verify kernels/<model>/<arch>/<op>.cu
```

This must exit zero before the PR is opened. `spite verify` runs the kernel against
the same inputs as the generic reference implementation and checks for numerical
agreement (absolute tolerance 1e-4 for f16 outputs, 1e-3 for Q8). A kernel that
produces wrong outputs at lower latency is not an improvement.

---

## PR format for kernel contributions

Title: `kernel: <model>/<arch> <operation>`
Example: `kernel: llama3/sm_86 attention`

Description sections (required):
1. **GPU** — exact card and driver version used for benchmarking
2. **Operation** — which op (attention / FFN / rms\_norm / MLA)
3. **Technique** — what makes this kernel faster (WMMA shapes, shared memory tiling, etc.)
4. **Before / After** — the benchmark table above
5. **Verify output** — paste the last line of `spite verify` output

---

## Rust host PRs

For changes to any `crates/spite-*` directory:

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test --workspace
```

All three must pass. If you add a new crate dependency, add it to `[workspace.dependencies]`
in the root `Cargo.toml` and reference it with `{ workspace = true }` in the crate's
`Cargo.toml` — do not pin a version only in the crate.

---

## ABI changes

`crates/spite-abi/src/lib.rs` is the contract between the Rust host and every compiled
kernel. If you change any `#[repr(C)]` type, any function pointer signature in
`SpiteKernelInfo`, or add/remove a field from `SpiteCtx`:

1. Increment `ABI_VERSION` (currently `3`).
2. List in the PR description which kernel `.so` files will silently break if not recompiled.
3. Kernel CI jobs run on the same PR to catch this automatically when path filters match.

Do not change `ABI_VERSION` for additive-only changes where old kernels still load and
run correctly (new optional fn pointer slot initialized to null is safe; removing a slot
is not).

---

## What agents may do

- Open PRs from their own branch or fork
- Push commits to their own branch (the one they opened the PR from)
- Run `cargo build`, `cargo test`, `cargo clippy`, `cargo fmt`
- Run `spite verify` and `spite bench`
- Use GitHub MCP tools to create PRs, add comments, and read CI results
- Add files to `kernels/<model>/<arch>/` without asking — the dispatcher picks them up

## What agents must not do

- Push to `main` or `master` directly
- Push to another contributor's branch without their explicit request
- Force-push to any branch
- Modify `kernels/<model>/<arch>/` files that belong to a different GPU architecture
- Change `ABI_VERSION` without a corresponding struct or signature change
- Submit a kernel PR without a `.bench` file and before/after numbers
- Disable or skip tests to make CI pass
