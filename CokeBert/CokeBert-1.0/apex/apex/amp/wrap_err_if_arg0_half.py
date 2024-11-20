def err_if_arg0_half(mod, fn, handle, verbose=False):
    if not utils.has_func(mod, fn):
        return
    orig_fn = utils.get_func(mod, fn)

    @functools.wraps(orig_fn)
    def wrapper(arg0, *args, **kwargs):
        assert compat.is_tensor_like(arg0)
        if utils.type_string(arg0) == 'HalfTensor':
            raise NotImplementedError('Cannot call in-place method ' +
                '{} on fp16 Tensors.'.format(fn))
        else:
            cast_fn = utils.verbosify(utils.maybe_float, fn, verbose)
            new_args = utils.casted_args(cast_fn, args, kwargs)
            return orig_fn(arg0, *new_args, **kwargs)
    utils.set_func_save(handle, mod, fn, wrapper)
