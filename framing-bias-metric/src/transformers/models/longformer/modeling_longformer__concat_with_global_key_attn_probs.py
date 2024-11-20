def _concat_with_global_key_attn_probs(self, key_vectors, query_vectors,
    max_num_global_attn_indices, is_index_global_attn_nonzero,
    is_local_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero):
    batch_size = key_vectors.shape[0]
    key_vectors_only_global = key_vectors.new_zeros(batch_size,
        max_num_global_attn_indices, self.num_heads, self.head_dim)
    key_vectors_only_global[is_local_index_global_attn_nonzero] = key_vectors[
        is_index_global_attn_nonzero]
    attn_probs_from_global_key = torch.einsum('blhd,bshd->blhs', (
        query_vectors, key_vectors_only_global))
    attn_probs_from_global_key[is_local_index_no_global_attn_nonzero[0], :,
        :, is_local_index_no_global_attn_nonzero[1]] = -10000.0
    return attn_probs_from_global_key
