def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = (apply_state or {}).get((var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)
    if self._momentum:
        momentum_var = self.get_slot(var, 'momentum')
        return training_ops.resource_apply_momentum(var.handle,
            momentum_var.handle, coefficients['lr_t'], grad, coefficients[
            'momentum'], use_locking=self._use_locking, use_nesterov=self.
            nesterov)
    else:
        return training_ops.resource_apply_gradient_descent(var.handle,
            coefficients['lr_t'], grad, use_locking=self._use_locking)
