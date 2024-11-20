def _prepare_local(self, var_device, var_dtype, apply_state):
    super(MomentumOptimizer, self)._prepare_local(var_device, var_dtype,
        apply_state)
    apply_state[var_device, var_dtype]['momentum'] = array_ops.identity(self
        ._get_hyper('momentum', var_dtype))
