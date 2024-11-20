def _linear(self, inputs):
    """
        Computes logits by running inputs through a linear layer.

        Args:
            inputs: A float32 tensor with shape [..., hidden_size]

        Returns:
            float32 tensor with shape [..., vocab_size].
        """
    first_dims = shape_list(inputs)[:-1]
    x = tf.reshape(inputs, [-1, self.hidden_size])
    logits = tf.matmul(x, self.weight, transpose_b=True)
    return tf.reshape(logits, first_dims + [self.vocab_size])
