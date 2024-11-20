def promote_function(fn):
    wrap_fn = functools.partial(wrap.make_promote_wrapper)
    return _decorator_helper(fn, utils.maybe_float, wrap_fn)
