def unshape(x):
    """  compute context """
    return tf.reshape(tf.transpose(x, perm=(0, 2, 1, 3)), (bs, -1, self.
        n_heads * dim_per_head))
