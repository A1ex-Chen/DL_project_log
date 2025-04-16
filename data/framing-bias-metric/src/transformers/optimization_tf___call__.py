def __call__(self, gradients):
    """Accumulates :obj:`gradients` on the current replica."""
    if not self._gradients:
        _ = self.step
        self._gradients.extend([(tf.Variable(tf.zeros_like(gradient),
            trainable=False, synchronization=tf.VariableSynchronization.
            ON_READ, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA) if
            gradient is not None else gradient) for gradient in gradients])
    if len(gradients) != len(self._gradients):
        raise ValueError('Expected %s gradients, but got %d' % (len(self.
            _gradients), len(gradients)))
    for accum_gradient, gradient in zip(self._gradients, gradients):
        if accum_gradient is not None and gradient is not None:
            accum_gradient.assign_add(gradient)
    self._accum_steps.assign_add(1)
