def build_dict(name, args=None):
    if name == 'ModelConfig':
        return_dict = copy.deepcopy(ModelConfig)
    elif name == 'BlockConfig':
        return_dict = copy.deepcopy(BlockConfig)
    else:
        raise ValueError('Name of requested dictionary not found!')
    if args is None:
        return return_dict
    if isinstance(args, dict):
        return_dict.update(args)
    elif isinstance(args, tuple):
        return_dict.update({a: p for a, p in zip(list(return_dict.keys()),
            args)})
    else:
        raise ValueError('Expected tuple or dict!')
    return return_dict
