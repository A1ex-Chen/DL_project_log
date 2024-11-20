def _reshape_and_cast(self, t):
    return tf.cast(tf.reshape(t, [-1]), tf.int32)
