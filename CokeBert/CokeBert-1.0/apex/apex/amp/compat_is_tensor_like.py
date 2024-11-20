def is_tensor_like(x):
    return torch.is_tensor(x) or isinstance(x, torch.autograd.Variable)
