@wraps(func)
def wrapper(*args, **kwargs):
    if is_tf_available():
        return func(*args, **kwargs)
    else:
        raise ImportError(f'Method `{func.__name__}` requires TF.')
