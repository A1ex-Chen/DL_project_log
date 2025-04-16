@staticmethod
def backward(ctx, grad_output):
    input_, mean, invvar = ctx.saved_tensors
    grad_input = None
    grad_input = fused_layer_norm_cuda.backward(grad_output.contiguous(),
        mean, invvar, input_, ctx.normalized_shape, ctx.eps)
    return grad_input, None, None
