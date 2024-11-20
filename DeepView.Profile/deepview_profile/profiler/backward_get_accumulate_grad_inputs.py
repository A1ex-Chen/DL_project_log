def get_accumulate_grad_inputs(initial_grad_fn, backward_runnable):
    input_dict = {}
    hook_handles = []

    def get_hook(grad_fn):

        def hook(arg1, arg2):
            if not isinstance(arg2[0], torch.Tensor):
                return
            input_dict[grad_fn] = arg2[0].size()
        return hook
    stack = [initial_grad_fn]
    visited = {initial_grad_fn}
    while len(stack) > 0:
        grad_fn = stack.pop()
        if grad_fn.name() == 'torch::autograd::AccumulateGrad':
            hook_handles.append(grad_fn.register_hook(get_hook(grad_fn)))
        for next_grad_fn, _ in grad_fn.next_functions:
            if next_grad_fn is None or next_grad_fn in visited:
                continue
            stack.append(next_grad_fn)
            visited.add(next_grad_fn)
    backward_runnable()
    torch.cuda.synchronize()
    for handle in hook_handles:
        handle.remove()
    return input_dict
