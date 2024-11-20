def _split_heads(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.attention_heads, int(self.
        attention_size / self.attention_heads)))
    return tf.transpose(x, perm=[0, 2, 1, 3])
