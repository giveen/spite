# Architecture

## Overview

spite is a Rust workspace. Each crate is one concern. Crates depend only
downward — nothing in the lower layers knows about the server or the CLI.

```
┌─────────────────────────────────────────────────────┐
│  spite-cli          Command-line interface           │
│  spite-server       OpenAI-compatible HTTP server    │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  spite-scheduler    Continuous batching              │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  spite-executor     Forward pass runner              │
│    spite-sampling   Token sampling (top-p, top-k)   │
│    spite-kvcache    KV cache + VBR quantization      │
│    spite-rope       Rotary positional embeddings     │
│    spite-lora       LoRA adapter injection           │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  spite-dispatch     Kernel selection at runtime      │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  spite-loader       GGUF file reader (mmap)          │
│  spite-models       Model weight-name maps           │
│  spite-tokenizer    Tokenize / detokenize            │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  spite-abi          C ABI contract (shared types)    │
└─────────────────────────────────────────────────────┘
```

---

## Data flow — one inference request

```
HTTP POST /v1/chat/completions
        │
        ▼
spite-server          parse JSON, validate request
        │
        ▼
spite-scheduler       assign a KV cache slot,
                      batch with other active requests
        │
        ▼
spite-executor        prefill(prompt_tokens)
  ├── spite-dispatch  → pick kernel .so for model × GPU
  ├── spite-kvcache   → populate KV cache for each layer
  └── spite-lora      → inject LoRA deltas if adapter loaded
        │
        ▼                ┌─────────────────────────┐
spite-executor        loop│  decode_step(token)     │
  ├── spite-dispatch  →   │  run single-token pass  │
  ├── spite-sampling  →   │  sample next token      │
  └── spite-kvcache   →   │  extend KV cache        │
                          └────────────── until EOS ┘
        │
        ▼
spite-tokenizer       detokenize tokens → text
        │
        ▼
spite-server          stream SSE chunks or return JSON
```

---

## Key crates

### spite-abi

The C ABI contract between the Rust host and C++23 GPU kernels. Defines
`SpiteCtx`, `SpiteTensor`, `SpiteKernelInfo`, and all function pointer
types (`AttentionFn`, `FfnFn`, `MlaFn`, etc.).

**Nothing outside spite-abi should define types that cross the FFI boundary.**
When you add a field to `SpiteCtx`, bump `ABI_VERSION`. Kernel `.so` files
built against the old version are rejected at load time.

Also defines `ShardStrategy` — lives here rather than spite-parallel to
avoid a dependency cycle.

### spite-loader

Opens a GGUF file, mmaps it, and exposes tensors by name. Zero allocation —
`SpiteTensor::data` points directly into the mmap buffer.

Parsing is in two parts:
- `lib.rs` — binary GGUF parsing (metadata KVs, tensor info, data offsets)
- `config.rs` — extract `ModelHyperparams` from the metadata map

### spite-dispatch

Loads kernel `.so` files at runtime using `libloading`. Selects the best
kernel for a given `(model_arch, gpu_arch)` pair using a priority chain:
exact match → model wildcard → gpu wildcard → generic fallback.

Kernels are hot-reloadable: the dispatcher watches for `.so` file changes
and swaps them in without restarting the server.

### spite-executor

Runs a forward pass. Owns the `ExecutorConfig` (context length, batch size,
thread count, KV quant config, offload policy). Currently stubbed — `prefill`
and `decode_step` return `ExecutorError::NotInitialized` until GPU kernel
plumbing is complete.

Also owns the `EngineRegistries` — pluggable samplers, tokenizers, and KV
cache backends resolved per-request by `PluginKey`.

### spite-scheduler

Continuous batching using the Orca / vLLM iteration-level approach. Each
request gets a `Slot` with its own KV cache region. At each step, all
`Generating` slots share one batched forward pass. New requests fill `Free`
slots without waiting for in-progress requests.

Owns the `Engine` (executor + registries).

### spite-kvcache

KV cache backends and quantization. Supports F32, F16, Q8, Q5_1, Q4.

Variable bit-rate (VBR) mode: the engine starts at F16 and degrades
automatically as VRAM fills — F16 → Q8 → Q5_1 → Q4. The `--kv-quant` flag
pins a starting tier; without it, degradation is automatic.

### spite-parallel

Tensor and pipeline parallelism for multi-GPU. Provides `gpu_aware_default()`
which selects `ShardStrategy::Tensor { n_shards }` when multiple GPUs are
present. `ShardStrategy` itself lives in spite-abi.

### spite-plugin

`Registry<T>` — a priority-chain plugin registry. Used by the executor to
hold samplers, tokenizers, and KV cache backends. A `PluginKey` with
`model_arch`, `gpu_arch`, and `task` fields is used to resolve the best
match: most-specific to least-specific.

### spite-offload

Weight offload policy: VRAM → System RAM → disk (mmap). Controls how weight
tensors are placed when VRAM is insufficient. `TieredPlacement::plan()`
takes model byte counts and available VRAM/RAM and returns a placement
decision with an eviction list.

Also handles vision encoder offload (`VisionSpec`, `VisionOffloadPolicy`):
encoders are placed in the best available tier and evicted after each image
prefill.

---

## Adding a new crate

```bash
cargo new --lib crates/spite-myfeature
```

Then add it to the workspace in `Cargo.toml`:

```toml
[workspace]
members = [
    # ...
    "crates/spite-myfeature",
]

[workspace.dependencies]
spite-myfeature = { path = "crates/spite-myfeature" }
```

Follow the naming convention: `spite-<noun>`. Keep the crate to one concern.
If a crate needs types from another, add a workspace dependency — never
copy types between crates.

---

## ABI versioning

`ABI_VERSION` in `spite-abi` is the enforcement mechanism for the C FFI.
The dispatcher calls `spite_kernel_info()` on every loaded `.so` and rejects
any kernel whose `abi_version` field doesn't match the host's `ABI_VERSION`.

Rules:
- Adding a new op to `SpiteKernelInfo` (as `Option<FnPtr>`) — bump version
- Adding a field to an existing `#[repr(C)]` struct — bump version, add a new struct instead if backwards compat matters
- Adding a new Rust-only type — no bump needed
- Renaming a Rust-only type — no bump needed

Kernels compiled against version N will not load against a host at version N+1.
This is intentional — it prevents silent corruption from mismatched ABIs.
