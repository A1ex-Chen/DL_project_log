def plain(module):
    if module.weight.grad is not None:
        return module.weight.grad * module.weight
    else:
        return torch.zeros_like(module.weight)
