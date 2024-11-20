def _decay_weights_op(self, var, learning_rate, apply_state):
    do_decay = self._do_use_weight_decay(var.name)
    if do_decay:
        return var.assign_sub(learning_rate * var * apply_state[var.device,
            var.dtype.base_dtype]['weight_decay_rate'], use_locking=self.
            _use_locking)
    return tf.no_op()
