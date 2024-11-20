@property
def step(self):
    """Number of accumulated steps."""
    if self._accum_steps is None:
        self._accum_steps = tf.Variable(tf.constant(0, dtype=tf.int64),
            trainable=False, synchronization=tf.VariableSynchronization.
            ON_READ, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
    return self._accum_steps.value()
