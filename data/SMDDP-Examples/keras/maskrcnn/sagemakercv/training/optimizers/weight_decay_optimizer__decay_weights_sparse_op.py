def _decay_weights_sparse_op(self, var, indices, apply_state=None):
    if not self._decay_var_list or var.ref() in self._decay_var_list:
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get((var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)
        update = -coefficients['wd_t'] * tf.gather(var, indices)
        return self._resource_scatter_add(var, indices, update)
    return tf.no_op()
