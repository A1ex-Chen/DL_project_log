def _has_inf_or_nan(x):
    try:
        cpu_sum = float(x.float().sum())
    except RuntimeError as instance:
        if 'value cannot be converted' not in instance.args[0]:
            raise
        return True
    else:
        if cpu_sum == float('inf') or cpu_sum == -float('inf'
            ) or cpu_sum != cpu_sum:
            return True
        return False
