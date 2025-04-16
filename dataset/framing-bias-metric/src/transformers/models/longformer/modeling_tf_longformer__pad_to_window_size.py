def _pad_to_window_size(self, input_ids, attention_mask, token_type_ids,
    position_ids, inputs_embeds, pad_token_id):
    """A helper function to pad tokens and mask to work with implementation of Longformer selfattention."""
    attention_window = self.attention_window if isinstance(self.
        attention_window, int) else max(self.attention_window)
    assert attention_window % 2 == 0, f'`attention_window` should be an even value. Given {attention_window}'
    input_shape = shape_list(input_ids
        ) if input_ids is not None else shape_list(inputs_embeds)
    batch_size, seq_len = input_shape[:2]
    padding_len = (attention_window - seq_len % attention_window
        ) % attention_window
    if padding_len > 0:
        logger.info(
            'Input ids are automatically padded from {} to {} to be a multiple of `config.attention_window`: {}'
            .format(seq_len, seq_len + padding_len, attention_window))
        paddings = tf.constant([[0, 0], [0, padding_len]])
        if input_ids is not None:
            input_ids = tf.pad(input_ids, paddings, constant_values=
                pad_token_id)
        if position_ids is not None:
            position_ids = tf.pad(position_ids, paddings, constant_values=
                pad_token_id)
        if inputs_embeds is not None:
            input_ids_padding = tf.fill((batch_size, padding_len), self.
                pad_token_id)
            inputs_embeds_padding = self.embeddings(input_ids_padding)
            inputs_embeds = tf.concat([inputs_embeds, inputs_embeds_padding
                ], axis=-2)
        attention_mask = tf.pad(attention_mask, paddings, constant_values=False
            )
        token_type_ids = tf.pad(token_type_ids, paddings, constant_values=0)
    return (padding_len, input_ids, attention_mask, token_type_ids,
        position_ids, inputs_embeds)
