@staticmethod
def _make_grid(nx=20, ny=20):
    xv, yv = tf.meshgrid(tf.range(nx), tf.range(ny))
    return tf.cast(tf.reshape(tf.stack([xv, yv], 2), [1, 1, ny * nx, 2]),
        dtype=tf.float32)
