def axpby_check_overflow_python(model_grad, stashed_grad, master_grad,
    scale, check_overflow=False):
    if check_overflow:
        cpu_sum = float(model_grad.float().sum())
        if cpu_sum == float('inf') or cpu_sum == -float('inf'
            ) or cpu_sum != cpu_sum:
            return True
    assert stashed_grad.dtype == master_grad.dtype
    converted_model_grad = model_grad.to(master_grad.dtype)
    stashed_grad.add_(scale, converted_model_grad)
    master_grad.data = stashed_grad.data
    return False
