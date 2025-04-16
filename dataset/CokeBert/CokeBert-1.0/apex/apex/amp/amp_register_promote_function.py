def register_promote_function(module, name):
    if not hasattr(module, name):
        raise ValueError('No function named {} in module {}.'.format(name,
            module))
    _USER_PROMOTE_REGISTRY.add((module, name))
