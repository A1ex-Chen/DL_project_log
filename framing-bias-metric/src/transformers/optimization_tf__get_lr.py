def _get_lr(self, var_device, var_dtype, apply_state):
    """Retrieves the learning rate with the given state."""
    if apply_state is None:
        return self._decayed_lr_t[var_dtype], {}
    apply_state = apply_state or {}
    coefficients = apply_state.get((var_device, var_dtype))
    if coefficients is None:
        coefficients = self._fallback_apply_state(var_device, var_dtype)
        apply_state[var_device, var_dtype] = coefficients
    return coefficients['lr_t'], dict(apply_state=apply_state)
