@classmethod
def new_from(cls, operation_outputs):
    retval, initial_grad_fn = get_grad_fn(operation_outputs)
    if initial_grad_fn is None:
        raise ValueError('No grad_fn available on the operation output.')
    grads = torch.ones_like(retval)

    def backward_runnable():
        torch.autograd.backward(retval, grads, retain_graph=True)
    size_dict = get_accumulate_grad_inputs(initial_grad_fn, backward_runnable)
    ag_dict = {grad_fn: torch.randn(size, device=torch.device('cuda')) for 
        grad_fn, size in size_dict.items()}
    return cls(backward_runnable, ag_dict)
