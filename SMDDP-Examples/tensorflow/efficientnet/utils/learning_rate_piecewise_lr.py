def piecewise_lr():
    return tf.compat.v1.train.piecewise_constant(tf.cast(step, tf.float32),
        self._step_boundaries, self._lr_values)
