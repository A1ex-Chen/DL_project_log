def prune(model, amount=0.3):
    import torch.nn.utils.prune as prune
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)
            prune.remove(m, 'weight')
    LOGGER.info(f'Model pruned to {sparsity(model):.3g} global sparsity')
