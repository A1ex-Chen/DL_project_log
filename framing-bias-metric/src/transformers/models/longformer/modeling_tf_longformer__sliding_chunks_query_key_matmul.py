def _sliding_chunks_query_key_matmul(self, query, key, window_overlap):
    """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This
        implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer) with an
        overlap of size window_overlap
        """
    batch_size, seq_len, num_heads, head_dim = shape_list(query)
    tf.debugging.assert_equal(seq_len % (window_overlap * 2), 0, message=
        f'Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}'
        )
    tf.debugging.assert_equal(shape_list(query), shape_list(key), message=
        f'Shape of query and key should be equal, but got query: {shape_list(query)} and key: {shape_list(key)}'
        )
    chunks_count = seq_len // window_overlap - 1
    query = tf.reshape(tf.transpose(query, (0, 2, 1, 3)), (batch_size *
        num_heads, seq_len, head_dim))
    key = tf.reshape(tf.transpose(key, (0, 2, 1, 3)), (batch_size *
        num_heads, seq_len, head_dim))
    chunked_query = self._chunk(query, window_overlap)
    chunked_key = self._chunk(key, window_overlap)
    chunked_attention_scores = tf.einsum('bcxd,bcyd->bcxy', chunked_query,
        chunked_key)
    paddings = tf.constant([[0, 0], [0, 0], [0, 1], [0, 0]], dtype=tf.
        dtypes.int32)
    diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
        chunked_attention_scores, paddings)
    diagonal_attn_scores_up_triang = tf.concat([
        diagonal_chunked_attention_scores[:, :, :window_overlap, :
        window_overlap + 1], diagonal_chunked_attention_scores[:, -1:,
        window_overlap:, :window_overlap + 1]], axis=1)
    diagonal_attn_scores_low_triang = tf.concat([tf.zeros((batch_size *
        num_heads, 1, window_overlap, window_overlap)),
        diagonal_chunked_attention_scores[:, :, -(window_overlap + 1):-1, 
        window_overlap + 1:]], axis=1)
    diagonal_attn_scores_first_chunk = tf.concat([tf.roll(
        diagonal_chunked_attention_scores, shift=[1, window_overlap], axis=
        [2, 3])[:, :, :window_overlap, :window_overlap], tf.zeros((
        batch_size * num_heads, 1, window_overlap, window_overlap))], axis=1)
    first_chunk_mask = tf.broadcast_to(tf.range(chunks_count + 1)[None, :,
        None, None], shape=(batch_size * num_heads, chunks_count + 1,
        window_overlap, window_overlap)) < 1
    diagonal_attn_scores_low_triang = tf.where(first_chunk_mask,
        diagonal_attn_scores_first_chunk, diagonal_attn_scores_low_triang)
    diagonal_attention_scores = tf.concat([diagonal_attn_scores_low_triang,
        diagonal_attn_scores_up_triang], axis=-1)
    diagonal_attention_scores = tf.transpose(tf.reshape(
        diagonal_attention_scores, (batch_size, num_heads, seq_len, 2 *
        window_overlap + 1)), (0, 2, 1, 3))
    diagonal_attention_scores = self._mask_invalid_locations(
        diagonal_attention_scores, window_overlap)
    return diagonal_attention_scores
