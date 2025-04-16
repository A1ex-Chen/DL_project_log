@wraps(func)
def run_in_eager_mode(*args, **kwargs):
    return func(*args, **kwargs)
