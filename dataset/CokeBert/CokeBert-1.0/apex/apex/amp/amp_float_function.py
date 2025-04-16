def float_function(fn):
    wrap_fn = functools.partial(wrap.make_cast_wrapper, try_caching=False)
    return _decorator_helper(fn, utils.maybe_float, wrap_fn)
