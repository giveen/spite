//! Bit-for-bit parity between the portable C KV block codec and the Rust host
//! codec.
//!
//! The block layouts are what make a quantized KV cache readable by a kernel,
//! so they exist in more than one language: `crates/spite-kvcache/src/quant.rs`
//! writes them on the host, `kernels/qwen/qwen3/nvidia/kv_attn.inl` reads them
//! on an NVIDIA device, and
//! `kernels/_engine/kv_quant/generic/kv_quant_blocks.h` is the portable
//! definition any other backend includes. If those drift apart, a kernel
//! silently decodes garbage, which is exactly the failure this test exists to
//! catch.
//!
//! It compiles `kv_quant_parity.c` against the portable header, runs it, and
//! compares both the encoded bytes and the decoded values with the Rust codec
//! over a shared deterministic corpus (half of it tiny enough that the block
//! scale is a subnormal half, where the two implementations previously
//! disagreed).
//!
//! Skips — loudly — when the machine has no C compiler to build the harness
//! with; the repo's other real-artifact tests skip the same way when the
//! hardware is absent.

use std::path::{Path, PathBuf};
use std::process::Command;

use spite_kvcache::{KvQuant, dequantize, packed_bytes, quantize};

const BLOCKS: usize = 64;
const ELEMS: usize = BLOCKS * 32;

/// Corpus shared with `kqp_corpus` in the C harness. Must stay identical or the
/// comparison is meaningless.
fn corpus(i: usize) -> f32 {
    let mut x = (i as u32).wrapping_mul(2_654_435_761);
    x ^= x >> 15;
    x = x.wrapping_mul(2_246_822_519);
    x ^= x >> 13;

    // Alternate magnitude per block: the small blocks put the block scale in
    // the f16 subnormal range.
    let scale = if (i / 32).is_multiple_of(2) {
        1e-3f32
    } else {
        1e-6f32
    };
    let mag = (x % 8192) as f32 * scale;
    if x & 0x8000_0000 != 0 { -mag } else { mag }
}

fn workspace_root() -> PathBuf {
    // .../crates/spite-kvcache -> the repo root.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crate lives two levels below the repo root")
        .to_path_buf()
}

fn find_compiler() -> Option<String> {
    let mut candidates: Vec<String> = Vec::new();
    if let Ok(cc) = std::env::var("CC")
        && !cc.is_empty()
    {
        candidates.push(cc);
    }
    candidates.extend(["cc", "gcc", "clang"].map(str::to_owned));

    candidates.into_iter().find(|c| {
        Command::new(c)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

fn hex_bytes(s: &str) -> Vec<u8> {
    assert!(s.len().is_multiple_of(2), "hex payload has odd length");
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).expect("hex payload"))
        .collect()
}

fn hex_u32s(s: &str) -> Vec<u32> {
    assert!(s.len().is_multiple_of(8), "u32 payload has an odd length");
    (0..s.len() / 8)
        .map(|i| u32::from_str_radix(&s[i * 8..i * 8 + 8], 16).expect("u32 payload"))
        .collect()
}

/// The tier each `E`/`D` line reports, mapped to the host enum.
fn tier_of(name: &str) -> KvQuant {
    match name {
        "q8_0" => KvQuant::Q8,
        "q5_1" => KvQuant::Q5_1,
        "q4_0" => KvQuant::Q4,
        other => panic!("harness reported an unknown tier {other:?}"),
    }
}

#[test]
fn portable_c_codec_matches_the_rust_codec() {
    let Some(cc) = find_compiler() else {
        eprintln!(
            "skipping: no C compiler (looked at $CC, cc, gcc, clang) — \
             cannot build the portable codec harness"
        );
        return;
    };

    let root = workspace_root();
    let src = root.join("kernels/_engine/kv_quant/generic/kv_quant_parity.c");
    assert!(
        src.is_file(),
        "portable codec harness missing at {}",
        src.display()
    );

    let out_dir = std::env::temp_dir().join(format!("spite-kvq-parity-{}", std::process::id()));
    std::fs::create_dir_all(&out_dir).expect("create temp dir");
    let exe = out_dir.join("kv_quant_parity");

    let build = Command::new(&cc)
        .arg("-std=c99")
        .arg("-O2")
        .arg("-I")
        .arg(&root)
        .arg(&src)
        .arg("-o")
        .arg(&exe)
        .arg("-lm") // roundf
        .output()
        .expect("run the C compiler");
    assert!(
        build.status.success(),
        "compiling the portable codec failed:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );

    let run = Command::new(&exe).output().expect("run the parity harness");
    assert!(run.status.success(), "parity harness exited {}", run.status);
    let stdout = String::from_utf8(run.stdout).expect("harness output is utf-8");

    let source: Vec<f32> = (0..ELEMS).map(corpus).collect();
    let mut tiers_seen = 0usize;

    for line in stdout.lines() {
        let mut parts = line.split(' ');
        let (which, tier_name, payload) = match (parts.next(), parts.next(), parts.next()) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => continue,
        };
        let q = tier_of(tier_name);
        let mut packed = vec![0u8; packed_bytes(q, ELEMS)];
        quantize(q, &source, &mut packed);

        match which {
            "E" => {
                let theirs = hex_bytes(payload);
                assert_eq!(
                    packed.len(),
                    theirs.len(),
                    "{tier_name}: harness encoded {} bytes, host allocates {}",
                    theirs.len(),
                    packed.len()
                );
                if let Some(i) = packed.iter().zip(&theirs).position(|(a, b)| a != b) {
                    panic!(
                        "{tier_name}: encoded byte {i} differs — port {:#04x} vs host {:#04x}",
                        theirs[i], packed[i]
                    );
                }
                tiers_seen += 1;
            }
            "D" => {
                let theirs = hex_u32s(payload);
                assert_eq!(theirs.len(), ELEMS, "{tier_name}: decoded element count");
                let mut back = vec![0f32; ELEMS];
                dequantize(q, &packed, ELEMS, &mut back);
                if let Some(i) = back
                    .iter()
                    .map(|v| v.to_bits())
                    .zip(&theirs)
                    .position(|(a, b)| a != *b)
                {
                    panic!(
                        "{tier_name}: decoded element {i} differs — port {} vs host {}",
                        f32::from_bits(theirs[i]),
                        f32::from_bits(back[i].to_bits())
                    );
                }
            }
            other => panic!("harness printed an unexpected record {other:?}"),
        }
    }

    assert_eq!(
        tiers_seen, 3,
        "expected encoded output for all three block tiers"
    );
}
