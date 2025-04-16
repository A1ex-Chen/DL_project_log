def setup_print_for_distributed(is_primary):
    """
    This function disables printing when not in primary process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_primary or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print
