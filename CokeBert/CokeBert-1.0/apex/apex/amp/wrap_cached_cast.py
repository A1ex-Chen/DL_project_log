def cached_cast(mod, fn, cast_fn, handle, try_caching=False, verbose=False):
    if not utils.has_func(mod, fn):
        return
    orig_fn = utils.get_func(mod, fn)
    cast_fn = utils.verbosify(cast_fn, fn, verbose)
    wrapper = make_cast_wrapper(orig_fn, cast_fn, handle, try_caching)
    utils.set_func_save(handle, mod, fn, wrapper)
