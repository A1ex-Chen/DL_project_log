def fisher(module):
    if module.fisher is not None:
        return torch.abs(module.fisher.detach())
    else:
        return torch.zeros(module.weight.shape[0])
