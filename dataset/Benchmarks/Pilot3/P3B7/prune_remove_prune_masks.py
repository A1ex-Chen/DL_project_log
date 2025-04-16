def remove_prune_masks(model: nn.Module):
    """Remove the `model` pruning masks.

    This is called after training with pruning so that
    we can save our model weights without the
    reparametrization of pruning.

    Args:
        model: model to be pruned

    Returns:
        model: model with pruning masks
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv1d):
            prune.remove(module, name='weight')
        elif isinstance(module, torch.nn.Linear):
            prune.remove(module, name='weight')
    return model
