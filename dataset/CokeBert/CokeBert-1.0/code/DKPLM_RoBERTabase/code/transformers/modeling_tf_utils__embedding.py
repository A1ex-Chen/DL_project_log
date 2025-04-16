def _embedding(self, input_ids):
    """Applies embedding based on inputs tensor."""
    return tf.gather(self.weight, input_ids)
