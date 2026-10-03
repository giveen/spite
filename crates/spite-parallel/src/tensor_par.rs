//! Megatron-style tensor (column/row) parallelism.

/// Column-parallel linear: each GPU holds columns [shard*cols/n .. (shard+1)*cols/n].
/// Forward: local matmul → all-reduce over shards.
pub struct ColParLinear {
    pub n_shards:   usize,
    pub shard_idx:  usize,
    pub in_features:  usize,
    pub out_features: usize, // total; local = out_features / n_shards
}

impl ColParLinear {
    /// Size of the local weight tile on this shard.
    pub fn local_out_features(&self) -> usize {
        let base  = self.out_features / self.n_shards;
        let extra = self.out_features % self.n_shards;
        base + if self.shard_idx < extra { 1 } else { 0 }
    }
}

/// Row-parallel linear: each GPU holds rows [shard*rows/n .. (shard+1)*rows/n].
/// Forward: local matmul → all-reduce (sum) over shards.
pub struct RowParLinear {
    pub n_shards:   usize,
    pub shard_idx:  usize,
    pub in_features:  usize, // total; local = in_features / n_shards
    pub out_features: usize,
}

impl RowParLinear {
    pub fn local_in_features(&self) -> usize {
        let base  = self.in_features / self.n_shards;
        let extra = self.in_features % self.n_shards;
        base + if self.shard_idx < extra { 1 } else { 0 }
    }
}
