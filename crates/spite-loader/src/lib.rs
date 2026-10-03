//! GGUF file loader.
//!
//! Opens a GGUF file, mmaps the tensor data, and exposes tensors by name.
//! No allocations are made for weight data — `SpiteTensor::data` points
//! directly into the mmap'd buffer for the lifetime of `GgufModel`.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use thiserror::Error;

use spite_abi::{SpiteTensor, SpiteType};

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a GGUF file (bad magic)")]
    BadMagic,
    #[error("unsupported GGUF version {0}")]
    UnsupportedVersion(u32),
    #[error("tensor not found: {0}")]
    TensorNotFound(String),
    #[error("unknown quant type {0}")]
    UnknownType(u32),
    #[error("malformed metadata key: {0}")]
    MalformedKey(String),
}

// ── GGUF constants ─────────────────────────────────────────────────────────

const GGUF_MAGIC:   u32 = 0x46554747; // "GGUF"
const GGUF_VERSION: u32 = 3;

// ── Internal tensor record ─────────────────────────────────────────────────

struct TensorRecord {
    offset: u64,
    ne:     [u32; 4],
    kind:   SpiteType,
}

// ── Public model handle ────────────────────────────────────────────────────

pub struct GgufModel {
    _file: File,
    mmap:  Mmap,
    meta:  HashMap<String, MetaValue>,
    tensors: HashMap<String, TensorRecord>,
    data_offset: u64,
}

impl GgufModel {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let file = File::open(path)?;
        // SAFETY: we hold the file open for the lifetime of mmap.
        let mmap = unsafe { Mmap::map(&file)? };

        let mut cursor = 0usize;

        let magic = read_u32(&mmap, &mut cursor);
        if magic != GGUF_MAGIC {
            return Err(LoadError::BadMagic);
        }

        let version = read_u32(&mmap, &mut cursor);
        if version != GGUF_VERSION {
            return Err(LoadError::UnsupportedVersion(version));
        }

        let n_tensors = read_u64(&mmap, &mut cursor) as usize;
        let n_kv      = read_u64(&mmap, &mut cursor) as usize;

        let mut meta = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let (key, value) = read_kv(&mmap, &mut cursor)?;
            meta.insert(key, value);
        }

        let mut tensors = HashMap::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let (name, record) = read_tensor_info(&mmap, &mut cursor)?;
            tensors.insert(name, record);
        }

        // tensor data starts at next 32-byte aligned offset after metadata
        let alignment = 32u64;
        let data_offset = (cursor as u64 + alignment - 1) / alignment * alignment;

        Ok(Self { _file: file, mmap, meta, tensors, data_offset })
    }

    // ── Metadata access ───────────────────────────────────────────────────

    pub fn arch(&self) -> &str {
        match self.meta.get("general.architecture") {
            Some(MetaValue::Str(s)) => s.as_str(),
            _ => "unknown",
        }
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.meta.get(key) {
            Some(MetaValue::U32(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.meta.get(key) {
            Some(MetaValue::F32(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.meta.get(key) {
            Some(MetaValue::Str(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    // ── Tensor access ─────────────────────────────────────────────────────

    /// Returns a tensor whose `data` pointer is valid for `'self` lifetime.
    /// Returns `SpiteTensor::null()` if the name doesn't exist.
    pub fn tensor(&self, name: &str) -> SpiteTensor {
        let Some(rec) = self.tensors.get(name) else {
            return SpiteTensor::null();
        };
        let ptr = unsafe {
            self.mmap.as_ptr().add((self.data_offset + rec.offset) as usize)
        };
        SpiteTensor {
            data: ptr as *mut _,
            ne:   rec.ne,
            kind: rec.kind,
        }
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    pub fn n_tensors(&self) -> usize {
        self.tensors.len()
    }
}

// ── GGUF parsing helpers ───────────────────────────────────────────────────

#[derive(Debug)]
enum MetaValue {
    U8(u8), I8(i8), U16(u16), I16(i16),
    U32(u32), I32(i32), U64(u64), I64(i64),
    F32(f32), F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<MetaValue>),
}

fn read_u8(buf: &[u8], cur: &mut usize) -> u8 {
    let v = buf[*cur]; *cur += 1; v
}
fn read_u16(buf: &[u8], cur: &mut usize) -> u16 {
    let v = u16::from_le_bytes(buf[*cur..*cur+2].try_into().unwrap());
    *cur += 2; v
}
fn read_u32(buf: &[u8], cur: &mut usize) -> u32 {
    let v = u32::from_le_bytes(buf[*cur..*cur+4].try_into().unwrap());
    *cur += 4; v
}
fn read_u64(buf: &[u8], cur: &mut usize) -> u64 {
    let v = u64::from_le_bytes(buf[*cur..*cur+8].try_into().unwrap());
    *cur += 8; v
}
fn read_f32(buf: &[u8], cur: &mut usize) -> f32 {
    f32::from_le_bytes(buf[*cur..*cur+4].try_into().unwrap()).also(|_| *cur += 4)
}
fn read_gguf_str(buf: &[u8], cur: &mut usize) -> String {
    let len = read_u64(buf, cur) as usize;
    let s = String::from_utf8_lossy(&buf[*cur..*cur+len]).into_owned();
    *cur += len; s
}

trait Also: Sized { fn also(self, f: impl FnOnce(&Self)) -> Self { f(&self); self } }
impl<T> Also for T {}

fn read_kv(buf: &[u8], cur: &mut usize) -> Result<(String, MetaValue), LoadError> {
    let key   = read_gguf_str(buf, cur);
    let vtype = read_u32(buf, cur);
    let value = read_meta_value(buf, cur, vtype)?;
    Ok((key, value))
}

fn read_meta_value(buf: &[u8], cur: &mut usize, vtype: u32) -> Result<MetaValue, LoadError> {
    Ok(match vtype {
        0  => MetaValue::U8  (read_u8(buf, cur)),
        1  => MetaValue::I8  (read_u8(buf, cur) as i8),
        2  => MetaValue::U16 (read_u16(buf, cur)),
        3  => MetaValue::I16 (read_u16(buf, cur) as i16),
        4  => MetaValue::U32 (read_u32(buf, cur)),
        5  => MetaValue::I32 (read_u32(buf, cur) as i32),
        6  => MetaValue::F32 (read_f32(buf, cur)),
        7  => MetaValue::Bool(read_u8(buf, cur) != 0),
        8  => MetaValue::Str (read_gguf_str(buf, cur)),
        9  => {
            let elem_type = read_u32(buf, cur);
            let count     = read_u64(buf, cur) as usize;
            let mut arr   = Vec::with_capacity(count);
            for _ in 0..count {
                arr.push(read_meta_value(buf, cur, elem_type)?);
            }
            MetaValue::Array(arr)
        }
        10 => MetaValue::U64(read_u64(buf, cur)),
        11 => MetaValue::I64(read_u64(buf, cur) as i64),
        12 => MetaValue::F64(f64::from_le_bytes(
            buf[*cur..*cur+8].try_into().unwrap()).also(|_| *cur += 8)),
        t  => return Err(LoadError::MalformedKey(format!("unknown value type {t}"))),
    })
}

fn read_tensor_info(buf: &[u8], cur: &mut usize) -> Result<(String, TensorRecord), LoadError> {
    let name  = read_gguf_str(buf, cur);
    let ndim  = read_u32(buf, cur) as usize;
    let mut ne = [1u32; 4];
    for i in 0..ndim {
        ne[i] = read_u64(buf, cur) as u32;
    }
    let type_id = read_u32(buf, cur);
    let offset  = read_u64(buf, cur);
    let kind = gguf_type(type_id)?;
    Ok((name, TensorRecord { offset, ne, kind }))
}

fn gguf_type(id: u32) -> Result<SpiteType, LoadError> {
    Ok(match id {
        0  => SpiteType::F32,
        1  => SpiteType::F16,
        2  => SpiteType::Q4_0,
        8  => SpiteType::Q8_0,
        10 => SpiteType::Q4K,
        11 => SpiteType::Q5K,
        12 => SpiteType::Q6K,
        30 => SpiteType::Bf16,
        t  => return Err(LoadError::UnknownType(t)),
    })
}
