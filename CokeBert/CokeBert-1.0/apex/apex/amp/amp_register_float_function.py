def register_float_function(module, name):
    if not hasattr(module, name):
        raise ValueError('No function named {} in module {}.'.format(name,
            module))
    _USER_CAST_REGISTRY.add((module, name, utils.maybe_float))
