def variable_is_tensor():
    v = torch.autograd.Variable()
    return isinstance(v, torch.Tensor)
