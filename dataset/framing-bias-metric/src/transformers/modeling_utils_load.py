def load(module: nn.Module, prefix=''):
    local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
    module._load_from_state_dict(state_dict, prefix, local_metadata, True,
        missing_keys, unexpected_keys, error_msgs)
    for name, child in module._modules.items():
        if child is not None:
            load(child, prefix + name + '.')
