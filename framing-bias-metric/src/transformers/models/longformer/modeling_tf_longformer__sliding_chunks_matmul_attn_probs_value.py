def _sliding_chunks_matmul_attn_probs_value(self, attn_probs, value,
    window_overlap):
    """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
    batch_size, seq_len, num_heads, head_dim = shape_list(value)
    tf.debugging.assert_equal(seq_len % (window_overlap * 2), 0, message=
        'Seq_len has to be multiple of 2 * window_overlap')
    tf.debugging.assert_equal(shape_list(attn_probs)[:3], shape_list(value)
        [:3], message=
        'value and attn_probs must have same dims (except head_dim)')
    tf.debugging.assert_equal(shape_list(attn_probs)[3], 2 * window_overlap +
        1, message='attn_probs last dim has to be 2 * window_overlap + 1')
    chunks_count = seq_len // window_overlap - 1
    chunked_attn_probs = tf.reshape(tf.transpose(attn_probs, (0, 2, 1, 3)),
        (batch_size * num_heads, seq_len // window_overlap, window_overlap,
        2 * window_overlap + 1))
    value = tf.reshape(tf.transpose(value, (0, 2, 1, 3)), (batch_size *
        num_heads, seq_len, head_dim))
    paddings = tf.constant([[0, 0], [window_overlap, window_overlap], [0, 0
        ]], dtype=tf.dtypes.int32)
    padded_value = tf.pad(value, paddings, constant_values=-1)
    frame_size = 3 * window_overlap * head_dim
    frame_hop_size = (shape_list(padded_value)[1] * head_dim - frame_size
        ) // chunks_count
    chunked_value = tf.signal.frame(tf.reshape(padded_value, (batch_size *
        num_heads, -1)), frame_size, frame_hop_size)
    chunked_value = tf.reshape(chunked_value, (batch_size * num_heads, 
        chunks_count + 1, 3 * window_overlap, head_dim))
    tf.debugging.assert_equal(shape_list(chunked_value), [batch_size *
        num_heads, chunks_count + 1, 3 * window_overlap, head_dim], message
        ='Chunked value has the wrong shape')
    chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)
    context = tf.einsum('bcwd,bcdh->bcwh', chunked_attn_probs, chunked_value)
    context = tf.transpose(tf.reshape(context, (batch_size, num_heads,
        seq_len, head_dim)), (0, 2, 1, 3))
    return context
