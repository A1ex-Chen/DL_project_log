def merge_heads(self, x):
    x = tf.transpose(x, [0, 2, 1, 3])
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2] + [x_shape[-2] * x_shape[-1]]
    return tf.reshape(x, new_x_shape)
