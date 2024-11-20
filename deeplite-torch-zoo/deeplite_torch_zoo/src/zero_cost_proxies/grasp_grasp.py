def grasp(module):
    if module.weight.grad is not None:
        return -module.weight.data * module.weight.grad
    else:
        return torch.zeros_like(module.weight)
