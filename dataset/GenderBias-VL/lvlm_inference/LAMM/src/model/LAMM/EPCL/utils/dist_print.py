def print(*args, **kwargs):
    force = kwargs.pop('force', False)
    if is_primary or force:
        builtin_print(*args, **kwargs)
