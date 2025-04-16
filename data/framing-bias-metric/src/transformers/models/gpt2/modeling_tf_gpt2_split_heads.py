def split_heads(self, x):
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-1] + [self.n_head, x_shape[-1] // self.n_head]
    x = tf.reshape(x, new_x_shape)
    return tf.transpose(x, (0, 2, 1, 3))
