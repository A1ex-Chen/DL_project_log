def applier(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    elif isinstance(value, string_classes):
        return value
    elif isinstance(value, np.ndarray):
        return value
    elif hasattr(value, 'to'):
        return fn(value)
    elif isinstance(value, container_abcs.Mapping):
        return {applier(k, fn): applier(v, fn) for k, v in value.items()}
    elif isinstance(value, container_abcs.Iterable):
        return type(value)(applier(v, fn) for v in value)
    else:
        return value
