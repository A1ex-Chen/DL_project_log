@tf.function
def update_average(self, step: tf.Tensor):
    step = tf.cast(step, tf.float32)
    if step < self._start_step:
        decay = tf.constant(0.0, tf.float32)
    elif self._dynamic_decay:
        decay = step - self._start_step
        decay = tf.minimum(self._average_decay, (1.0 + decay) / (10.0 + decay))
    else:
        decay = self._average_decay

    def _apply_moving(v_moving, v_normal):
        diff = v_moving - v_normal
        v_moving.assign_sub(tf.cast(1.0 - decay, v_moving.dtype) * diff)
        return v_moving

    def _update(strategy, v_moving_and_v_normal):
        for v_moving, v_normal in v_moving_and_v_normal:
            strategy.extended.update(v_moving, _apply_moving, args=(v_normal,))
    ctx = tf.distribute.get_replica_context()
    return ctx.merge_call(_update, args=(zip(self._average_weights, self.
        _model_weights),))
