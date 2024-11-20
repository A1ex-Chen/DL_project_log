def _compute_attn_output_with_global_indices(self, value_vectors,
    attn_probs, max_num_global_attn_indices, is_index_global_attn_nonzero,
    is_local_index_global_attn_nonzero):
    batch_size = shape_list(attn_probs)[0]
    attn_probs_only_global = attn_probs[:, :, :, :max_num_global_attn_indices]
    global_value_vectors = tf.gather_nd(value_vectors,
        is_index_global_attn_nonzero)
    value_vectors_only_global = tf.scatter_nd(
        is_local_index_global_attn_nonzero, global_value_vectors, shape=(
        batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim)
        )
    attn_output_only_global = tf.einsum('blhs,bshd->blhd',
        attn_probs_only_global, value_vectors_only_global)
    attn_probs_without_global = attn_probs[:, :, :,
        max_num_global_attn_indices:]
    attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(
        attn_probs_without_global, value_vectors, self.
        one_sided_attn_window_size)
    return attn_output_only_global + attn_output_without_global
