@staticmethod
def backward(ctx, out_grad1):
    g1, = ctx.saved_tensors
    return g1 * out_grad1, None, None
