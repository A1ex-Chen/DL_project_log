@functools.wraps(func)
def new_func(*args, **kwargs):
    warnings.simplefilter('always', DeprecationWarning)
    warnings.warn(f'Method {func.__name__} is deprecated.', category=
        DeprecationWarning, stacklevel=2)
    warnings.simplefilter('default', DeprecationWarning)
    return func(*args, **kwargs)