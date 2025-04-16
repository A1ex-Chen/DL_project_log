def is_floating_point(x):
    if hasattr(torch, 'is_floating_point'):
        return torch.is_floating_point(x)
    try:
        torch_type = x.type()
        return torch_type.endswith('FloatTensor') or torch_type.endswith(
            'HalfTensor') or torch_type.endswith('DoubleTensor')
    except AttributeError:
        return False
