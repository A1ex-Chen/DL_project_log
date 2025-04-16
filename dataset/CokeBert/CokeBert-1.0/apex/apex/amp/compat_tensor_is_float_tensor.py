def tensor_is_float_tensor():
    x = torch.Tensor()
    return type(x) == torch.FloatTensor
