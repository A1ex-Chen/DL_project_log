def reset(self):
    """Resets the accumulated gradients on the current replica."""
    if not self._gradients:
        return
    self._accum_steps.assign(0)
    for gradient in self._gradients:
        if gradient is not None:
            gradient.assign(tf.zeros_like(gradient))
