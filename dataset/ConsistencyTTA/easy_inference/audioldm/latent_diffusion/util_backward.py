@staticmethod
def backward(ctx, *output_grads):
    ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.
        input_tensors]
    with torch.enable_grad():
        shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
        output_tensors = ctx.run_function(*shallow_copies)
    input_grads = torch.autograd.grad(output_tensors, ctx.input_tensors +
        ctx.input_params, output_grads, allow_unused=True)
    del ctx.input_tensors
    del ctx.input_params
    del output_tensors
    return (None, None) + input_grads
