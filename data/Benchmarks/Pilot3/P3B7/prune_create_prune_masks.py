def create_prune_masks(model: nn.Module):
    """Update the `model` with pruning masks.

    Args:
        model: model to be pruned

    Returns:
        model: model with pruning masks
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv1d):
            prune.l1_unstructured(module, name='weight', amount=0.4)
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.2)
    return model
