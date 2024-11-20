def get_extended_attention_mask(self, attention_mask, input_shape, dtype):
    if attention_mask is None:
        attention_mask = tf.fill(input_shape, 1)
    extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
    extended_attention_mask = tf.cast(extended_attention_mask, dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask
