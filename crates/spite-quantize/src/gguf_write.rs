//! Minimal GGUF v3 writer, used by [`crate::quantize_model`].
//!
//! Only what the quantizer needs: re-emit the source metadata verbatim (from the
//! loader's parsed `MetaValue` map), then a new tensor index and data section.
//! Metadata keys are written in sorted order so runs are reproducible.

use std::collections::HashMap;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

use spite_loader::MetaValue;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF"
const GGUF_VERSION: u32 = 3;
const ALIGN: u64 = 32;

/// One tensor to write: name, shape, GGUF type id, packed bytes.
pub struct OutTensor {
    pub name: String,
    pub ne: [u32; 4],
    /// Original GGUF rank, preserved so [n,1] does not become [n].
    pub ndim: u32,
    pub type_id: u32,
    pub bytes: Vec<u8>,
}

fn put_str<W: Write>(w: &mut W, s: &str) -> std::io::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())
}

fn meta_type_id(v: &MetaValue) -> u32 {
    match v {
        MetaValue::U8(_) => 0,
        MetaValue::I8(_) => 1,
        MetaValue::U16(_) => 2,
        MetaValue::I16(_) => 3,
        MetaValue::U32(_) => 4,
        MetaValue::I32(_) => 5,
        MetaValue::F32(_) => 6,
        MetaValue::Bool(_) => 7,
        MetaValue::Str(_) => 8,
        MetaValue::Array(_) => 9,
        MetaValue::U64(_) => 10,
        MetaValue::I64(_) => 11,
        MetaValue::F64(_) => 12,
    }
}

fn put_meta_value<W: Write>(w: &mut W, v: &MetaValue) -> std::io::Result<()> {
    match v {
        MetaValue::U8(x) => w.write_all(&[*x]),
        MetaValue::I8(x) => w.write_all(&[*x as u8]),
        MetaValue::U16(x) => w.write_all(&x.to_le_bytes()),
        MetaValue::I16(x) => w.write_all(&x.to_le_bytes()),
        MetaValue::U32(x) => w.write_all(&x.to_le_bytes()),
        MetaValue::I32(x) => w.write_all(&x.to_le_bytes()),
        MetaValue::F32(x) => w.write_all(&x.to_le_bytes()),
        MetaValue::Bool(x) => w.write_all(&[u8::from(*x)]),
        MetaValue::Str(s) => put_str(w, s),
        MetaValue::U64(x) => w.write_all(&x.to_le_bytes()),
        MetaValue::I64(x) => w.write_all(&x.to_le_bytes()),
        MetaValue::F64(x) => w.write_all(&x.to_le_bytes()),
        MetaValue::Array(items) => {
            // GGUF arrays are homogeneous; take the element type from the first
            // entry (empty arrays default to U32 — no known file has one).
            let et = items.first().map(meta_type_id).unwrap_or(4);
            w.write_all(&et.to_le_bytes())?;
            w.write_all(&(items.len() as u64).to_le_bytes())?;
            for it in items {
                put_meta_value(w, it)?;
            }
            Ok(())
        }
    }
}

/// Write `tensors` (in the given order) with `meta`, into a GGUF v3 file.
pub fn write(
    path: &Path,
    meta: &HashMap<String, MetaValue>,
    tensors: &[OutTensor],
) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    w.write_all(&GGUF_MAGIC.to_le_bytes())?;
    w.write_all(&GGUF_VERSION.to_le_bytes())?;
    w.write_all(&(tensors.len() as u64).to_le_bytes())?;
    w.write_all(&(meta.len() as u64).to_le_bytes())?;

    let mut keys: Vec<&String> = meta.keys().collect();
    keys.sort();
    for k in keys {
        let v = &meta[k];
        put_str(&mut w, k)?;
        w.write_all(&meta_type_id(v).to_le_bytes())?;
        put_meta_value(&mut w, v)?;
    }

    // Data offsets, each tensor aligned to 32 bytes from the data-section start.
    let mut offset: u64 = 0;
    let mut offsets = Vec::with_capacity(tensors.len());
    for t in tensors {
        offset = offset.div_ceil(ALIGN) * ALIGN;
        offsets.push(offset);
        offset += t.bytes.len() as u64;
    }

    for (t, &off) in tensors.iter().zip(&offsets) {
        put_str(&mut w, &t.name)?;
        let ndim = t.ndim.clamp(1, 4) as usize;
        w.write_all(&(ndim as u32).to_le_bytes())?;
        for d in &t.ne[..ndim] {
            w.write_all(&(*d as u64).to_le_bytes())?;
        }
        w.write_all(&t.type_id.to_le_bytes())?;
        w.write_all(&off.to_le_bytes())?;
    }

    // Pad to the data-section alignment, then emit each tensor at its offset.
    let pos = w.stream_position()?;
    let data_start = pos.div_ceil(ALIGN) * ALIGN;
    for _ in pos..data_start {
        w.write_all(&[0u8])?;
    }
    let mut written = 0u64;
    for (t, &off) in tensors.iter().zip(&offsets) {
        while written < off {
            w.write_all(&[0u8])?;
            written += 1;
        }
        w.write_all(&t.bytes)?;
        written += t.bytes.len() as u64;
    }
    w.flush()
}
