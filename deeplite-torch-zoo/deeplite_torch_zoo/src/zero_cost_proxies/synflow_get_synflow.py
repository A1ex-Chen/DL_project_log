def get_synflow(layer):
    if layer.weight.grad is not None:
        return torch.abs(layer.weight * layer.weight.grad)
    else:
        return torch.zeros_like(layer.weight)
