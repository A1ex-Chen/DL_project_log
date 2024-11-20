def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))
