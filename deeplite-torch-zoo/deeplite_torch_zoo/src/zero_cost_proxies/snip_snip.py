def snip(module):
    if module.weight_mask.grad is not None:
        return torch.abs(module.weight_mask.grad)
    else:
        return torch.zeros_like(module.weight)
