@staticmethod
def backward(ctx, grad):
    shape = ctx.shape
    return _NewEmptyTensorOp.apply(grad, shape), None
