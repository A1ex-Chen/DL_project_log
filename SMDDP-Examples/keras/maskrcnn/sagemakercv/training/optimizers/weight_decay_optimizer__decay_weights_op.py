def _decay_weights_op(self, var, apply_state=None):
    if not self._decay_var_list or var.ref() in self._decay_var_list:
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get((var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)
        return var.assign_sub(coefficients['wd_t'] * var, self._use_locking)
    return tf.no_op()
