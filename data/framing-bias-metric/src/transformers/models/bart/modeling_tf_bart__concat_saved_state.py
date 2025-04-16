def _concat_saved_state(self, k, v, saved_state, static_kv, bsz) ->Tuple[tf
    .Tensor]:
    prev_key = tf.reshape(saved_state['prev_key'], (bsz * self.num_heads, -
        1, self.head_dim))
    k = prev_key if static_kv else tf.concat([prev_key, k], axis=1)
    prev_value = tf.reshape(saved_state['prev_value'], (bsz * self.
        num_heads, -1, self.head_dim))
    v = prev_value if static_kv else tf.concat([prev_value, v], axis=1)
    return k, v
