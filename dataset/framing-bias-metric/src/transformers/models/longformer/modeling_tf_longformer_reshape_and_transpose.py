def reshape_and_transpose(self, vector, batch_size):
    return tf.reshape(tf.transpose(tf.reshape(vector, (batch_size, -1, self
        .num_heads, self.head_dim)), (0, 2, 1, 3)), (batch_size * self.
        num_heads, -1, self.head_dim))
