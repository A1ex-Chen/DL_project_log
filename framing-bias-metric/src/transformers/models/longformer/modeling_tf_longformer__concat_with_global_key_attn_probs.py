def _concat_with_global_key_attn_probs(self, attn_scores, key_vectors,
    query_vectors, max_num_global_attn_indices,
    is_index_global_attn_nonzero, is_local_index_global_attn_nonzero,
    is_local_index_no_global_attn_nonzero):
    batch_size = shape_list(key_vectors)[0]
    global_key_vectors = tf.gather_nd(key_vectors, is_index_global_attn_nonzero
        )
    key_vectors_only_global = tf.scatter_nd(is_local_index_global_attn_nonzero,
        global_key_vectors, shape=(batch_size, max_num_global_attn_indices,
        self.num_heads, self.head_dim))
    attn_probs_from_global_key = tf.einsum('blhd,bshd->blhs', query_vectors,
        key_vectors_only_global)
    attn_probs_from_global_key_trans = tf.transpose(attn_probs_from_global_key,
        (0, 3, 1, 2))
    mask_shape = (shape_list(is_local_index_no_global_attn_nonzero)[0],
        ) + tuple(shape_list(attn_probs_from_global_key_trans)[-2:])
    mask = tf.ones(mask_shape) * -10000.0
    attn_probs_from_global_key_trans = tf.tensor_scatter_nd_update(
        attn_probs_from_global_key_trans,
        is_local_index_no_global_attn_nonzero, mask)
    attn_probs_from_global_key = tf.transpose(attn_probs_from_global_key_trans,
        (0, 2, 3, 1))
    attn_scores = tf.concat((attn_probs_from_global_key, attn_scores), axis=-1)
    return attn_scores
