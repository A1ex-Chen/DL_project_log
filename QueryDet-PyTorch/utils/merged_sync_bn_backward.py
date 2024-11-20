@staticmethod
def backward(ctx, grad_output):
    dist.all_reduce(grad_output, async_op=False)
    return grad_output
