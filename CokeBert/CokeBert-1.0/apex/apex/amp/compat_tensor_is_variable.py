def tensor_is_variable():
    x = torch.Tensor()
    return type(x) == torch.autograd.Variable
