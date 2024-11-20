def prepare_inputs_for_generation(self, inputs, **kwargs):
    mask_token_id = self.config.mask_token_id
    lang_id = self.config.lang_id
    effective_batch_size = inputs.shape[0]
    mask_token = tf.ones((effective_batch_size, 1), dtype=tf.int32
        ) * mask_token_id
    inputs = tf.concat([inputs, mask_token], axis=1)
    if lang_id is not None:
        langs = tf.ones_like(inputs) * lang_id
    else:
        langs = None
    return {'input_ids': inputs, 'langs': langs}
