def transpose_for_scores(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.
        attention_head_size))
    return tf.transpose(x, perm=[0, 2, 1, 3])
