def patch_tensor_constructor(fn):

    def wrapper(*args, **kwargs):
        kwargs['device'] = device
        return fn(*args, **kwargs)
    return wrapper
