def _shift_right(self, input_ids):
    decoder_start_token_id = self.config.decoder_start_token_id
    pad_token_id = self.config.pad_token_id
    assert decoder_start_token_id is not None, 'self.model.config.decoder_start_token_id has to be defined. In TF T5 it is usually set to the pad_token_id. See T5 docs for more information'
    shifted_input_ids = tf.cast(input_ids, tf.int32)
    shifted_input_ids = tf.roll(shifted_input_ids, 1, axis=-1)
    start_tokens = tf.fill((shape_list(shifted_input_ids)[0], 1),
        decoder_start_token_id)
    shifted_input_ids = tf.concat([start_tokens, shifted_input_ids[:, 1:]], -1)
    assert pad_token_id is not None, 'self.model.config.pad_token_id has to be defined.'
    shifted_input_ids = tf.where(shifted_input_ids == -100, tf.fill(
        shape_list(shifted_input_ids), pad_token_id), shifted_input_ids)
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.
        cast(0, tf.int32))
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)
    return shifted_input_ids
