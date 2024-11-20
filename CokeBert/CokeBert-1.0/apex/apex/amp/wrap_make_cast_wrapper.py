def make_cast_wrapper(orig_fn, cast_fn, handle, try_caching=False):

    @functools.wraps(orig_fn)
    def wrapper(*args, **kwargs):
        if not handle.is_active():
            return orig_fn(*args, **kwargs)
        if try_caching and handle.has_cache:
            args = list(args)
            for i in range(len(args)):
                if utils.should_cache(args[i]):
                    args[i] = utils.cached_cast(cast_fn, args[i], handle.cache)
            for k in kwargs:
                if utils.should_cache(kwargs[k]):
                    kwargs[k] = utils.cached_cast(cast_fn, kwargs[k],
                        handle.cache)
        new_args = utils.casted_args(cast_fn, args, kwargs)
        return orig_fn(*new_args, **kwargs)
    return wrapper
