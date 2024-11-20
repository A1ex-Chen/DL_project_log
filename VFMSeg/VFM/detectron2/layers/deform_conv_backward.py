@staticmethod
@once_differentiable
def backward(ctx, grad_output):
    if not grad_output.is_cuda:
        raise NotImplementedError('Deformable Conv is not supported on CPUs!')
    input, offset, mask, weight, bias = ctx.saved_tensors
    grad_input = torch.zeros_like(input)
    grad_offset = torch.zeros_like(offset)
    grad_mask = torch.zeros_like(mask)
    grad_weight = torch.zeros_like(weight)
    grad_bias = torch.zeros_like(bias)
    _C.modulated_deform_conv_backward(input, weight, bias, ctx._bufs[0],
        offset, mask, ctx._bufs[1], grad_input, grad_weight, grad_bias,
        grad_offset, grad_mask, grad_output, weight.shape[2], weight.shape[
        3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation,
        ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
    if not ctx.with_bias:
        grad_bias = None
    return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
        None, None, None, None, None)
