@staticmethod
def forward(ctx, x):
    ctx.save_for_backward(x)
    return x.mul(torch.tanh(F.softplus(x)))
