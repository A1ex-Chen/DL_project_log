def patch_FSDP_use_orig_params(func):

    def wrap_func(*args, **kwargs):
        use_orig_params = kwargs.pop('use_orig_params', True)
        return func(*args, **kwargs, use_orig_params=use_orig_params)
    return wrap_func
