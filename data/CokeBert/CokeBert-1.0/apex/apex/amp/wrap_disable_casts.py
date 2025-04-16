def disable_casts(mod, fn, handle):
    if not utils.has_func(mod, fn):
        return
    orig_fn = utils.get_func(mod, fn)

    @functools.wraps(orig_fn)
    def wrapper(*args, **kwargs):
        with handle._disable_casts():
            return orig_fn(*args, **kwargs)
    utils.set_func_save(handle, mod, fn, wrapper)
