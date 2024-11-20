def _compute_global_attn_output_from_hidden(self, attn_output,
    hidden_states, max_num_global_attn_indices,
    is_local_index_global_attn_nonzero, is_index_global_attn_nonzero,
    is_local_index_no_global_attn_nonzero, is_index_masked, training):
    batch_size, seq_len = shape_list(hidden_states)[:2]
    global_attn_hidden_states = tf.gather_nd(hidden_states,
        is_index_global_attn_nonzero)
    global_attn_hidden_states = tf.scatter_nd(
        is_local_index_global_attn_nonzero, global_attn_hidden_states,
        shape=(batch_size, max_num_global_attn_indices, self.embed_dim))
    global_query_vectors_only_global = self.query_global(
        global_attn_hidden_states)
    global_key_vectors = self.key_global(hidden_states)
    global_value_vectors = self.value_global(hidden_states)
    global_query_vectors_only_global /= tf.math.sqrt(tf.constant(self.
        head_dim, dtype=tf.dtypes.float32))
    global_query_vectors_only_global = self.reshape_and_transpose(
        global_query_vectors_only_global, batch_size)
    global_key_vectors = self.reshape_and_transpose(global_key_vectors,
        batch_size)
    global_value_vectors = self.reshape_and_transpose(global_value_vectors,
        batch_size)
    global_attn_scores = tf.matmul(global_query_vectors_only_global,
        global_key_vectors, transpose_b=True)
    tf.debugging.assert_equal(shape_list(global_attn_scores), [batch_size *
        self.num_heads, max_num_global_attn_indices, seq_len], message=
        f'global_attn_scores have the wrong size. Size should be {batch_size * self.num_heads, max_num_global_attn_indices, seq_len}, but is {shape_list(global_attn_scores)}.'
        )
    global_attn_scores = tf.reshape(global_attn_scores, (batch_size, self.
        num_heads, max_num_global_attn_indices, seq_len))
    global_attn_scores_trans = tf.transpose(global_attn_scores, (0, 2, 1, 3))
    mask_shape = (shape_list(is_local_index_no_global_attn_nonzero)[0],
        ) + tuple(shape_list(global_attn_scores_trans)[-2:])
    global_attn_mask = tf.ones(mask_shape) * -10000.0
    global_attn_scores_trans = tf.tensor_scatter_nd_update(
        global_attn_scores_trans, is_local_index_no_global_attn_nonzero,
        global_attn_mask)
    global_attn_scores = tf.transpose(global_attn_scores_trans, (0, 2, 1, 3))
    attn_mask = tf.broadcast_to(is_index_masked[:, None, None, :],
        shape_list(global_attn_scores))
    global_attn_scores = tf.where(attn_mask, -10000.0, global_attn_scores)
    global_attn_scores = tf.reshape(global_attn_scores, (batch_size * self.
        num_heads, max_num_global_attn_indices, seq_len))
    global_attn_probs_float = tf.nn.softmax(global_attn_scores, axis=-1)
    global_attn_probs = self.global_dropout(global_attn_probs_float,
        training=training)
    global_attn_output = tf.matmul(global_attn_probs, global_value_vectors)
    tf.debugging.assert_equal(shape_list(global_attn_output), [batch_size *
        self.num_heads, max_num_global_attn_indices, self.head_dim],
        message=
        f'global_attn_output tensor has the wrong size. Size should be {batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim}, but is {shape_list(global_attn_output)}.'
        )
    global_attn_output = tf.reshape(global_attn_output, (batch_size, self.
        num_heads, max_num_global_attn_indices, self.head_dim))
    nonzero_global_attn_output = tf.gather_nd(tf.transpose(
        global_attn_output, (0, 2, 1, 3)), is_local_index_global_attn_nonzero)
    nonzero_global_attn_output = tf.reshape(nonzero_global_attn_output, (
        shape_list(is_local_index_global_attn_nonzero)[0], -1))
    attn_output = tf.tensor_scatter_nd_update(attn_output,
        is_index_global_attn_nonzero, nonzero_global_attn_output)
    global_attn_probs = tf.reshape(global_attn_probs, (batch_size, self.
        num_heads, max_num_global_attn_indices, seq_len))
    return attn_output, global_attn_probs
