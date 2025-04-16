def _decorator_helper(orig_fn, cast_fn, wrap_fn):

    def wrapper(*args, **kwargs):
        handle = _DECORATOR_HANDLE
        if handle is None or not handle.is_active():
            return orig_fn(*args, **kwargs)
        inner_cast_fn = utils.verbosify(cast_fn, orig_fn.__name__, handle.
            verbose)
        return wrap_fn(orig_fn, inner_cast_fn, handle)(*args, **kwargs)
    return wrapper
