def _gen_VF_wrapper(name):

    def wrapper(*args, **kwargs):
        return getattr(_VF, name)(*args, **kwargs)
    return wrapper
