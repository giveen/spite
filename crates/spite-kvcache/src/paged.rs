//! Paged KV cache (PagedAttention-style).
//!
//! The key insight from PagedAttention (Kwon et al. 2023): instead of
//! pre-allocating a contiguous KV buffer of size `max_seq_len`, allocate
//! fixed-size *pages* of `block_size` token slots and map them to sequences
//! via a per-sequence block table. This:
//!
//! - Eliminates internal fragmentation (no reserved-but-unused context space)
//! - Allows sharing common prefix pages across requests (copy-on-write)
//! - Makes KV memory a pool usable by the scheduler across all active requests
//!
//! # Layout
//!
//! A physical block holds `block_size` token positions for all layers at once
//! (one tensor per layer per block). The block table maps `seq_id → [block_id]`.
//!
//! In production this would use GPU memory; here we use `Vec<u8>` as a stand-in.

use crate::CacheError;

/// Size of one page in token positions.
pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// A single physical KV block — `block_size` slots × all layers × K and V.
pub struct KvBlock {
    pub block_id: usize,
    /// Number of token positions currently written (0..=block_size).
    pub filled:   usize,
    /// Reference count: how many sequences share this block.
    pub ref_count: usize,
    // TODO: replace with GPU allocation handle (cudaMalloc etc.)
    _data: Vec<u8>,
}

impl KvBlock {
    fn new(block_id: usize, bytes: usize) -> Self {
        Self {
            block_id,
            filled: 0,
            ref_count: 0,
            _data: vec![0u8; bytes],
        }
    }

    pub fn is_full(&self, block_size: usize) -> bool {
        self.filled >= block_size
    }
}

/// Pool of all physical KV blocks.
pub struct BlockPool {
    block_size: usize,
    n_layers:   usize,
    n_kv_heads: usize,
    head_dim:   usize,
    blocks:     Vec<KvBlock>,
    /// Stack of free block ids.
    free:       Vec<usize>,
}

impl BlockPool {
    /// Allocate `n_blocks` physical blocks.
    pub fn new(
        n_blocks:   usize,
        block_size: usize,
        n_layers:   usize,
        n_kv_heads: usize,
        head_dim:   usize,
    ) -> Self {
        let bytes_per_block = block_size * n_layers * n_kv_heads * head_dim * 2 /* fp16 */ * 2 /* K+V */;
        let mut blocks = Vec::with_capacity(n_blocks);
        let mut free   = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            blocks.push(KvBlock::new(i, bytes_per_block));
            free.push(i);
        }
        Self { block_size, n_layers, n_kv_heads, head_dim, blocks, free }
    }

    pub fn n_free(&self) -> usize { self.free.len() }
    pub fn n_total(&self) -> usize { self.blocks.len() }

    /// Allocate one free block; returns its id.
    pub fn alloc(&mut self) -> Result<usize, CacheError> {
        self.free.pop().ok_or_else(|| CacheError::Alloc("block pool exhausted".into()))
    }

    /// Return a block to the pool (unconditional — caller manages ref counts).
    pub fn free_block(&mut self, id: usize) {
        self.blocks[id].filled    = 0;
        self.blocks[id].ref_count = 0;
        self.free.push(id);
    }
}

/// Per-sequence paged KV cache: a dynamic list of block ids.
pub struct PagedSeqCache {
    pub seq_id:     u64,
    block_table:    Vec<usize>,
    /// Total token positions written (across all blocks).
    n_tokens:       usize,
}

impl PagedSeqCache {
    pub fn new(seq_id: u64) -> Self {
        Self { seq_id, block_table: Vec::new(), n_tokens: 0 }
    }

    pub fn n_tokens(&self) -> usize { self.n_tokens }

    /// Number of blocks currently allocated for this sequence.
    pub fn n_blocks(&self) -> usize { self.block_table.len() }

    /// Ensure the next token position is covered; allocate a new block if needed.
    pub fn prepare_next(&mut self, pool: &mut BlockPool) -> Result<(), CacheError> {
        let need_new_block = self.block_table.is_empty()
            || pool.blocks[*self.block_table.last().unwrap()].is_full(pool.block_size);
        if need_new_block {
            let id = pool.alloc()?;
            pool.blocks[id].ref_count += 1;
            self.block_table.push(id);
        }
        Ok(())
    }

    /// Commit the last write head forward by one token.
    pub fn commit(&mut self, pool: &mut BlockPool) {
        if let Some(&last) = self.block_table.last() {
            pool.blocks[last].filled += 1;
        }
        self.n_tokens += 1;
    }

    /// Free all blocks back to the pool.
    pub fn free_all(&mut self, pool: &mut BlockPool) {
        for &id in &self.block_table {
            let block = &mut pool.blocks[id];
            block.ref_count -= 1;
            if block.ref_count == 0 {
                pool.free_block(id);
            }
        }
        self.block_table.clear();
        self.n_tokens = 0;
    }

    /// Read-only view of the block table (for the attention kernel).
    pub fn block_table(&self) -> &[usize] {
        &self.block_table
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_free() {
        let mut pool = BlockPool::new(8, DEFAULT_BLOCK_SIZE, 4, 8, 128);
        assert_eq!(pool.n_free(), 8);

        let mut seq = PagedSeqCache::new(1);
        for _ in 0..DEFAULT_BLOCK_SIZE {
            seq.prepare_next(&mut pool).unwrap();
            seq.commit(&mut pool);
        }
        // One full block used.
        assert_eq!(seq.n_blocks(), 1);
        assert_eq!(seq.n_tokens(), DEFAULT_BLOCK_SIZE);
        assert_eq!(pool.n_free(), 7);

        seq.free_all(&mut pool);
        assert_eq!(pool.n_free(), 8);
    }
}
