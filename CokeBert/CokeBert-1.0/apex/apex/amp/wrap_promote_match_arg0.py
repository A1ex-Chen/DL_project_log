def promote_match_arg0(mod, fn, handle, verbose=False):
    if not utils.has_func(mod, fn):
        return
    orig_fn = utils.get_func(mod, fn)

    @functools.wraps(orig_fn)
    def wrapper(arg0, *args, **kwargs):
        assert compat.is_tensor_like(arg0)
        if not _amp_state.handle.is_active():
            return orig_fn(arg0, *args, **kwargs)
        if utils.type_string(arg0) == 'HalfTensor':
            cast_fn = utils.maybe_half
        elif utils.type_string(arg0) == 'FloatTensor':
            cast_fn = utils.maybe_float
        else:
            return orig_fn(arg0, *args, **kwargs)
        cast_fn = utils.verbosify(cast_fn, fn, verbose)
        new_args = utils.casted_args(cast_fn, args, kwargs)
        return orig_fn(arg0, *new_args, **kwargs)
    utils.set_func_save(handle, mod, fn, wrapper)
