def _get_args_from_config(from_config_func, *args, **kwargs):
    """
    Use `from_config` to obtain explicit arguments.

    Returns:
        dict: arguments to be used for cls.__init__
    """
    signature = inspect.signature(from_config_func)
    if list(signature.parameters.keys())[0] != 'cfg':
        if inspect.isfunction(from_config_func):
            name = from_config_func.__name__
        else:
            name = f'{from_config_func.__self__}.from_config'
        raise TypeError(f"{name} must take 'cfg' as the first argument!")
    support_var_arg = any(param.kind in [param.VAR_POSITIONAL, param.
        VAR_KEYWORD] for param in signature.parameters.values())
    if support_var_arg:
        ret = from_config_func(*args, **kwargs)
    else:
        supported_arg_names = set(signature.parameters.keys())
        extra_kwargs = {}
        for name in list(kwargs.keys()):
            if name not in supported_arg_names:
                extra_kwargs[name] = kwargs.pop(name)
        ret = from_config_func(*args, **kwargs)
        ret.update(extra_kwargs)
    return ret
