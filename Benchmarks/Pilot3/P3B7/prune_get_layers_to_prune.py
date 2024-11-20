def get_layers_to_prune(model: nn.Module):
    """Get layers to be pruned"""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv1d):
            layers.append((module, 'weight'))
        elif isinstance(module, torch.nn.Linear):
            layers.append((module, 'weight'))
            print(f'Pruning {module}')
    return layers
