@functools.wraps(orig_func)
def wrapped(*args, **kwargs):
    if _called_with_cfg(*args, **kwargs):
        explicit_args = _get_args_from_config(from_config, *args, **kwargs)
        return orig_func(**explicit_args)
    else:
        return orig_func(*args, **kwargs)
