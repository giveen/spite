//! Token tree speculation (Medusa / EAGLE style).
//!
//! Instead of a linear draft sequence, multiple heads predict several
//! possible continuations simultaneously, forming a tree. The main model
//! verifies all branches in one batched forward pass using a tree
//! attention mask.
//!
//! Example with 3 heads, each predicting top-2:
//!
//!   pos 0:  [A, B]
//!   pos 1:  [AA, AB, BA, BB]
//!   pos 2:  [AAA, AAB, ABA, ABB, ...]
//!
//! The main model runs once over all candidates with a sparse attention
//! mask that prevents each candidate from attending to incompatible
//! branch siblings.
//!
//! This requires a `tree_attention` kernel op — a future addition to the
//! ABI. Medusa-style heads are additional parameters baked into the model
//! weights, not a separate model.

/// A node in the speculation tree.
pub struct TreeNode {
    pub token_id: u32,
    pub logit: f32,
    pub children: Vec<TreeNode>,
    pub depth: usize,
}

/// Build a draft tree from `n_heads` prediction heads, each keeping `top_k`
/// candidates.
pub fn build_tree(
    _logits_per_head: &[Vec<f32>], // [n_heads][vocab_size]
    _top_k: usize,
) -> Vec<TreeNode> {
    // TODO: for each head, take top-k logits and build tree level
    vec![]
}

/// Flatten the tree into a sequence for batched verification,
/// alongside the attention mask that encodes tree structure.
///
/// Returns:
///   tokens:  flattened candidate tokens [total_nodes]
///   mask:    bool [total_nodes, total_nodes] — which positions can attend
pub fn flatten_tree(_tree: &[TreeNode]) -> (Vec<u32>, Vec<Vec<bool>>) {
    // TODO: BFS flatten + build causal-with-tree mask
    (vec![], vec![])
}

/// Walk the verified accept mask and find the longest accepted path
/// in the tree.
pub fn accepted_path(_tree: &[TreeNode], _accept_mask: &[bool]) -> Vec<u32> {
    // TODO: greedy longest-path walk from root following accepted nodes
    vec![]
}
