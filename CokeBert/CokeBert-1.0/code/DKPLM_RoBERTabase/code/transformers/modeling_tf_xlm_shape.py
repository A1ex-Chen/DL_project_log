def shape(x):
    """  projection """
    return tf.transpose(tf.reshape(x, (bs, -1, self.n_heads, dim_per_head)),
        perm=(0, 2, 1, 3))
