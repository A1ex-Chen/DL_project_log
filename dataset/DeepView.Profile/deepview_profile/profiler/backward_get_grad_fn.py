def get_grad_fn(retval):
    if isinstance(retval, torch.Tensor) and retval.grad_fn is not None:
        return retval, retval.grad_fn
    elif isinstance(retval, tuple) or isinstance(retval, list):
        for inner_value in retval:
            inner_retval, grad_fn = get_grad_fn(inner_value)
            if grad_fn is not None:
                return inner_retval, grad_fn
    return None, None
