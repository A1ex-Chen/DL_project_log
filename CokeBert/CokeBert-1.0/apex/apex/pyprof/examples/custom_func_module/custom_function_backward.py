@staticmethod
def backward(ctx, grad):
    in1_grad = grad
    in2_grad = grad
    return in1_grad, in2_grad
