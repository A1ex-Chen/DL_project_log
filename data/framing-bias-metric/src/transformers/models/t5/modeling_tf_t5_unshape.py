def unshape(hidden_states):
    """  compute context """
    return tf.reshape(tf.transpose(hidden_states, perm=(0, 2, 1, 3)), (
        batch_size, -1, self.inner_dim))
