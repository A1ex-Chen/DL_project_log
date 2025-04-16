def has_overflow(self, params):
    for p in params:
        if p.grad is not None and DynamicLossScaler._has_inf_or_nan(p.grad.data
            ):
            return True
    return False
