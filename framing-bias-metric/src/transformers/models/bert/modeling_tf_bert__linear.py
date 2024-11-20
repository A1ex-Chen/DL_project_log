def _linear(self, inputs):
    """
        Computes logits by running inputs through a linear layer.

        Args:
            inputs: A float32 tensor with shape [batch_size, length, hidden_size].

        Returns:
            float32 tensor with shape [batch_size, length, vocab_size].
        """
    batch_size = shape_list(inputs)[0]
    length = shape_list(inputs)[1]
    x = tf.reshape(inputs, [-1, self.hidden_size])
    logits = tf.matmul(x, self.word_embeddings, transpose_b=True)
    return tf.reshape(logits, [batch_size, length, self.vocab_size])
