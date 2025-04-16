def _compute_attn_output_with_global_indices(self, value_vectors,
    attn_probs, max_num_global_attn_indices, is_index_global_attn_nonzero,
    is_local_index_global_attn_nonzero):
    batch_size = attn_probs.shape[0]
    attn_probs_only_global = attn_probs.narrow(-1, 0,
        max_num_global_attn_indices)
    value_vectors_only_global = value_vectors.new_zeros(batch_size,
        max_num_global_attn_indices, self.num_heads, self.head_dim)
    value_vectors_only_global[is_local_index_global_attn_nonzero
        ] = value_vectors[is_index_global_attn_nonzero]
    attn_output_only_global = torch.matmul(attn_probs_only_global.transpose
        (1, 2), value_vectors_only_global.transpose(1, 2)).transpose(1, 2)
    attn_probs_without_global = attn_probs.narrow(-1,
        max_num_global_attn_indices, attn_probs.size(-1) -
        max_num_global_attn_indices).contiguous()
    attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(
        attn_probs_without_global, value_vectors, self.
        one_sided_attn_window_size)
    return attn_output_only_global + attn_output_without_global
