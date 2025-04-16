def load(module: torch.nn.Module, prefix=''):
    args = state_dict, prefix, {}, True, [], [], error_msgs
    module._load_from_state_dict(*args)
    for name, child in module._modules.items():
        if child is not None:
            load(child, prefix + name + '.')
