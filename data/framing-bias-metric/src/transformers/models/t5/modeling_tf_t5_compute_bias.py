def compute_bias(self, query_length, key_length):
    """ Compute binned relative position bias """
    context_position = tf.range(query_length)[:, None]
    memory_position = tf.range(key_length)[None, :]
    relative_position = memory_position - context_position
    relative_position_bucket = self._relative_position_bucket(relative_position
        , bidirectional=not self.is_decoder, num_buckets=self.
        relative_attention_num_buckets)
    values = self.relative_attention_bias(relative_position_bucket)
    values = tf.expand_dims(tf.transpose(values, [2, 0, 1]), axis=0)
    return values
