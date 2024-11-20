@staticmethod
def backward(ctx, grad_output):
    return grad_output * ctx.scale, None
