def _resource_apply_sparse_duplicate_indices(self, grad, var, indices, **kwargs
    ):
    if self._momentum:
        return super(MomentumOptimizer, self
            )._resource_apply_sparse_duplicate_indices(grad, var, indices,
            **kwargs)
    else:
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = kwargs.get('apply_state', {}).get((var_device,
            var_dtype)) or self._fallback_apply_state(var_device, var_dtype)
        return resource_variable_ops.resource_scatter_add(var.handle,
            indices, -grad * coefficients['lr_t'])
