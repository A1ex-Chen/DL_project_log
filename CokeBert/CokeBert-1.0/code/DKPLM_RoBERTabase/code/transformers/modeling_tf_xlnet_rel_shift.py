@staticmethod
def rel_shift(x, klen=-1):
    """perform relative shift to form the relative attention score."""
    x_size = shape_list(x)
    x = tf.reshape(x, (x_size[1], x_size[0], x_size[2], x_size[3]))
    x = x[1:, ...]
    x = tf.reshape(x, (x_size[0], x_size[1] - 1, x_size[2], x_size[3]))
    x = x[:, 0:klen, :, :]
    return x
