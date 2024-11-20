def scale_check_overflow_python(model_grad, master_grad, scale,
    check_overflow=False):
    if check_overflow:
        cpu_sum = float(model_grad.float().sum())
        if cpu_sum == float('inf') or cpu_sum == -float('inf'
            ) or cpu_sum != cpu_sum:
            return True
    if master_grad is not model_grad:
        master_grad.copy_(model_grad)
    if scale != 1.0:
        master_grad.mul_(scale)
    return False
