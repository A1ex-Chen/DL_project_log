def _compute_global_attention_mask(input_ids_shape, sep_token_indices,
    before_sep_token=True):
    """
    Computes global attention mask by putting attention on all tokens before `sep_token_id` if `before_sep_token is
    True` else after `sep_token_id`.
    """
    assert sep_token_indices.shape[1
        ] == 2, '`input_ids` should have two dimensions'
    question_end_index = tf.reshape(sep_token_indices, (input_ids_shape[0],
        3, 2))[:, 0, 1]
    question_end_index = tf.cast(question_end_index[:, None], tf.dtypes.int32)
    attention_mask = tf.range(input_ids_shape[1])
    if before_sep_token is True:
        attention_mask = tf.cast(tf.broadcast_to(attention_mask,
            input_ids_shape) < tf.broadcast_to(question_end_index,
            input_ids_shape), tf.dtypes.int32)
    else:
        attention_mask = tf.cast(tf.broadcast_to(attention_mask,
            input_ids_shape) > tf.broadcast_to(question_end_index + 1,
            input_ids_shape), tf.dtypes.int32) * tf.cast(tf.broadcast_to(
            attention_mask, input_ids_shape) < input_ids_shape[-1], tf.
            dtypes.int32)
    return attention_mask
