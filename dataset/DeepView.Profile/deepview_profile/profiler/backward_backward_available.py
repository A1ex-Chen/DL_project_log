def backward_available(operation_output):
    return get_grad_fn(operation_output)[1] is not None
