def shape(hidden_states):
    """  projection """
    return tf.transpose(tf.reshape(hidden_states, (batch_size, -1, self.
        n_heads, self.key_value_proj_dim)), perm=(0, 2, 1, 3))
