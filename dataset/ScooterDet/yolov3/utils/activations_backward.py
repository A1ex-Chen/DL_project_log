@staticmethod
def backward(ctx, grad_output):
    x = ctx.saved_tensors[0]
    sx = torch.sigmoid(x)
    fx = F.softplus(x).tanh()
    return grad_output * (fx + x * sx * (1 - fx * fx))
