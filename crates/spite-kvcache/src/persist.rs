//! KV cache persistence — save/restore cache state to disk.
//!
//! Allows "prompt caching": run the prefill for a long system prompt once,
//! save the resulting KV state, and restore it for every new conversation
//! without paying the prefill cost again.
//!
//! File format (.spkv):
//!   [0..4]   magic  0x5350_4B56  ("SPKV")
//!   [4..8]   version  u32 = 1
//!   [8..]    CacheHeader (packed)
//!   [header_end..] raw KV tensor bytes for all layers

use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};
use thiserror::Error;

pub const CACHE_MAGIC: u32 = 0x5350_4B56; // "SPKV"
pub const CACHE_VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum PersistError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("model mismatch: saved for {saved:?}, loaded {current:?}")]
    ModelMismatch { saved: String, current: String },
    #[error("unsupported cache version {0}")]
    VersionMismatch(u32),
    #[error("truncated cache file")]
    Truncated,
    #[error("bad magic bytes")]
    BadMagic,
}

/// Metadata header stored at the start of every .spkv file.
#[derive(Debug, Clone)]
pub struct CacheHeader {
    pub n_layers: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub seq_len: u32,   // number of prefix tokens whose KV is stored
    pub arch_hash: u64, // FNV-1a hash of arch name + config for mismatch detection
}

impl CacheHeader {
    pub fn byte_size(&self) -> usize {
        let per_layer = 2
            * self.n_kv_heads as usize
            * self.head_dim as usize
            * self.seq_len as usize
            * std::mem::size_of::<f32>();
        per_layer * self.n_layers as usize
    }
}

/// Save a KV cache snapshot.
///
/// `kv_data` must be exactly `header.byte_size()` bytes.
pub fn save(path: &Path, header: &CacheHeader, kv_data: &[u8]) -> Result<(), PersistError> {
    let mut f = File::create(path)?;
    f.write_all(&CACHE_MAGIC.to_le_bytes())?;
    f.write_all(&CACHE_VERSION.to_le_bytes())?;
    f.write_all(&header.n_layers.to_le_bytes())?;
    f.write_all(&header.n_kv_heads.to_le_bytes())?;
    f.write_all(&header.head_dim.to_le_bytes())?;
    f.write_all(&header.seq_len.to_le_bytes())?;
    f.write_all(&header.arch_hash.to_le_bytes())?;
    f.write_all(kv_data)?;
    Ok(())
}

/// Load a KV cache snapshot.
///
/// Validates magic and version; `kv_data` is filled with the tensor bytes.
pub fn load(path: &Path, kv_data: &mut Vec<u8>) -> Result<CacheHeader, PersistError> {
    let mut f = File::open(path)?;
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    f.read_exact(&mut buf4)?;
    if u32::from_le_bytes(buf4) != CACHE_MAGIC {
        return Err(PersistError::BadMagic);
    }

    f.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version != CACHE_VERSION {
        return Err(PersistError::VersionMismatch(version));
    }

    let mut u32_field = || -> Result<u32, PersistError> {
        f.read_exact(&mut buf4)?;
        Ok(u32::from_le_bytes(buf4))
    };

    let header = CacheHeader {
        n_layers: u32_field()?,
        n_kv_heads: u32_field()?,
        head_dim: u32_field()?,
        seq_len: u32_field()?,
        arch_hash: {
            f.read_exact(&mut buf8)?;
            u64::from_le_bytes(buf8)
        },
    };

    let expected = header.byte_size();
    kv_data.resize(expected, 0);
    f.read_exact(kv_data).map_err(|_| PersistError::Truncated)?;
    Ok(header)
}
