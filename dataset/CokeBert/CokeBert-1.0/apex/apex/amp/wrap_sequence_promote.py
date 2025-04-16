def sequence_promote(mod, fn, handle, verbose=False):
    orig_fn = utils.get_func(mod, fn)
    maybe_float = utils.verbosify(utils.maybe_float, fn, verbose)

    @functools.wraps(orig_fn)
    def wrapper(seq, *args, **kwargs):
        if not _amp_state.handle.is_active():
            return orig_fn(seq, *args, **kwargs)
        types = set([utils.type_string(x) for x in seq])
        if len(types) <= 1:
            return orig_fn(seq, *args, **kwargs)
        elif types == set(['HalfTensor', 'FloatTensor']):
            cast_seq = utils.casted_args(maybe_float, seq, {})
            return orig_fn(cast_seq, *args, **kwargs)
        else:
            return orig_fn(seq, *args, **kwargs)
    utils.set_func_save(handle, mod, fn, wrapper)
