def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = (apply_state or {}).get((var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)
    momentum_var = self.get_slot(var, 'momentum')
    return training_ops.resource_sparse_apply_momentum(var.handle,
        momentum_var.handle, coefficients['lr_t'], grad, indices,
        coefficients['momentum'], use_locking=self._use_locking,
        use_nesterov=self.nesterov)
