# Testing

## Philosophy

spite separates two concerns:

1. **Plumbing tests** — does the Rust code compile, load, route, and wire
   together correctly? Runs on any laptop or CI runner. No GPU needed. Fast.
2. **Kernel tests** — does this specific kernel produce correct output on
   this specific GPU? Runs only on hardware that matches the kernel. Not
   required for merging Rust-only changes.

Every PR must pass the plumbing tests. Kernel tests are required only for
PRs that add or modify kernel files.

---

## Fake GGUF models

Real GGUF models are gigabytes. Tests can't download them in CI. Instead,
`spite-testkit` generates minimal valid GGUF files at test time:

```rust
use spite_testkit::FakeGguf;

let tmp = FakeGguf::default().write_to_tempfile().unwrap();
let model = GgufModel::open(tmp.path()).unwrap();
assert_eq!(model.arch(), "llama");
```

The fake file is structurally identical to a real GGUF — the loader reads
the same code path — but dimensions are tiny (`d_model = 64`, `n_layers = 2`)
and weight tensors contain zeros. The whole file is a few kilobytes.

### Configuring fake models

```rust
let fake = FakeGguf {
    arch:      "mistral".to_string(),
    n_layers:  4,
    d_model:   128,
    d_ffn:     512,
    n_heads:   4,
    n_kv_heads: 2,
    vocab_size: 64,
    ..Default::default()
};
let tmp = fake.write_to_tempfile().unwrap();
```

---

## Running tests locally

```bash
# All Rust tests (no GPU needed)
cargo test --workspace

# One specific crate
cargo test -p spite-loader

# One specific test
cargo test -p spite-loader hyperparams_parse

# With output visible
cargo test -p spite-loader -- --nocapture
```

---

## Test layout

```
crates/spite-testkit/
  src/lib.rs              FakeGguf writer + self-tests

crates/spite-loader/
  tests/
    loader_integration.rs  loader open/read/tensor/hyperparams tests

crates/spite-server/
  tests/
    api_integration.rs     HTTP endpoint tests (health, models, dispatch)
```

Each crate also has `#[cfg(test)]` unit tests inline in `src/`.

---

## Adding a new test

### Unit test (in a single function)

Add a `#[test]` block inside the module it belongs to:

```rust
// crates/spite-kvcache/src/lib.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eviction_frees_slot() {
        // ...
    }
}
```

### Integration test (touches multiple crates)

Create or add to `tests/` in the relevant crate. Use `FakeGguf` when
you need a model. Add `spite-testkit` to that crate's `[dev-dependencies]`:

```toml
[dev-dependencies]
spite-testkit = { workspace = true }
```

### Async test (HTTP, I/O)

Use `#[tokio::test]` and add `tokio` to dev-dependencies:

```toml
[dev-dependencies]
tokio = { workspace = true, features = ["full"] }
```

---

## Kernel tests (hardware required)

Kernel tests run `spite verify` and `spite bench` against real hardware.
They are not in the Rust test suite — they are CMake test targets:

```bash
# Build kernels for your GPU
cmake -B build -DSPITE_GPU_ARCHS="sm_89" && cmake --build build

# Verify correctness
spite verify kernels/llama3/sm_89/attention.cu

# Benchmark
spite bench kernels/llama3/sm_89/attention.cu
```

These run only on self-hosted CI runners tagged with the matching GPU
architecture. See `.github/workflows/ci.yml` for which jobs are required
vs. optional.

---

## CI overview

| Job              | Runs on             | Required | Triggers              |
|------------------|---------------------|----------|-----------------------|
| `rust`           | ubuntu-latest       | yes      | every PR              |
| `kernels-cuda`   | self-hosted, cuda   | no       | `kernels/**/sm_*/**`  |
| `kernels-hip`    | self-hosted, hip    | no       | `kernels/**/rdna*/**` |
| `kernels-metal`  | self-hosted, metal  | no       | `kernels/**/metal/**` |

The `rust` job is the merge gate. GPU jobs are advisory — they run when
available and report results, but a kernel PR without a matching runner
merges with a note that hardware testing didn't run.
