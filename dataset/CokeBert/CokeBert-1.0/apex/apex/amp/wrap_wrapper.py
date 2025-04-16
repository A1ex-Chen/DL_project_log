@functools.wraps(orig_fn)
def wrapper(*args, **kwargs):
    with handle._disable_casts():
        return orig_fn(*args, **kwargs)
