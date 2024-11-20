@staticmethod
def backward(ctx, grad_output):
    if ctx.scale > 1:
        mask, = ctx.saved_tensors
        return grad_output.masked_fill(mask, 0) * ctx.scale, None
    else:
        return grad_output, None
