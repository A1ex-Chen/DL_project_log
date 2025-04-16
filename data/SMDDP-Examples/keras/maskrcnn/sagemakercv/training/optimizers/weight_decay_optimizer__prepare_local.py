def _prepare_local(self, var_device, var_dtype, apply_state):
    super(DecoupledWeightDecayExtension, self)._prepare_local(var_device,
        var_dtype, apply_state)
    if 'weight_decay' in self._hyper:
        wd_t = tf.identity(self._decayed_wd(var_dtype))
        apply_state[var_device, var_dtype]['wd_t'] = wd_t
