def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))
