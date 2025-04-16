def _extract_gradient_functions_in_topological_order(model_output):
    """
    Given a model output (Tensor or nested list/tuple of Tensors), build a
    topological ordering of their gradient functions.
    """
    if isinstance(model_output, tuple) or isinstance(model_output, list):
        tensors = _flatten_and_filter_tensors(model_output)
    elif isinstance(model_output, torch.Tensor
        ) and model_output.grad_fn is not None:
        tensors = [model_output]
    else:
        return []
    result = []
    visited = {tensor.grad_fn for tensor in tensors}
    stack = [(grad_fn, 0) for grad_fn in visited]
    while len(stack) > 0:
        grad_fn, visit_count = stack.pop()
        if visit_count != 0:
            result.append(grad_fn)
            continue
        stack.append((grad_fn, 1))
        for fn, _ in grad_fn.next_functions:
            if fn is None or fn in visited:
                continue
            visited.add(fn)
            stack.append((fn, 0))
    result.reverse()
    return result
