def torch_required(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_torch_available():
            return func(*args, **kwargs)
        else:
            raise ImportError(f'Method `{func.__name__}` requires PyTorch.')
    return wrapper
