def _prepare_local(self, var_device, var_dtype, apply_state):
    super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype,
        apply_state)
    apply_state[var_device, var_dtype]['weight_decay_rate'] = tf.constant(self
        .weight_decay_rate, name='adam_weight_decay_rate')
