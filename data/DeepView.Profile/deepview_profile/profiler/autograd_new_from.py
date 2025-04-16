@classmethod
def new_from(cls, operation_output, exclude_accumulate_grad=True):
    _, initial_grad_fn = get_grad_fn(operation_output)
    if initial_grad_fn is None:
        raise ValueError('No grad_fn available on the operation output.')
    ordering = []
    input_map = {}
    initial_inputs = [tensor.detach() for tensor in
        flatten_operation_output(operation_output)]
    input_map[initial_grad_fn] = len(initial_inputs)
    stack = [(initial_grad_fn, 0)]
    visited = {initial_grad_fn}
    while len(stack) > 0:
        grad_fn, visit_count = stack.pop()
        if visit_count != 0:
            ordering.append(grad_fn)
            continue
        stack.append((grad_fn, 1))
        for next_fn, input_idx in grad_fn.next_functions:
            if next_fn is None:
                continue
            if exclude_accumulate_grad and next_fn.name(
                ) == 'torch::autograd::AccumulateGrad':
                continue
            if next_fn not in input_map:
                input_map[next_fn] = 1
            input_map[next_fn] = max(input_map[next_fn], input_idx + 1)
            if next_fn in visited:
                continue
            visited.add(next_fn)
            stack.append((next_fn, 0))
    ordering.reverse()
    return cls(ordering, input_map, initial_inputs)
