def __call__(self, step: int):
    """Compute learning rate at given step."""

    def warmup_lr():
        return self._rescaled_lr * (step / tf.cast(self._warmup_steps, tf.
            float32))

    def piecewise_lr():
        return tf.compat.v1.train.piecewise_constant(tf.cast(step, tf.
            float32), self._step_boundaries, self._lr_values)
    return tf.cond(step < self._warmup_steps, warmup_lr, piecewise_lr)
