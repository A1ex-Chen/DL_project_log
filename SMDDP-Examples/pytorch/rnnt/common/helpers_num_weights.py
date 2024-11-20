def num_weights(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
