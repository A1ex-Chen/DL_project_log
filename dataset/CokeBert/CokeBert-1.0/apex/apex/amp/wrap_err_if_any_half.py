def err_if_any_half(mod, fn, handle, custom_err_msg=None):
    if not utils.has_func(mod, fn):
        return
    orig_fn = utils.get_func(mod, fn)

    @functools.wraps(orig_fn)
    def wrapper(*args, **kwargs):
        types = utils.collect_fp_tensor_types(args, kwargs)
        if 'HalfTensor' in types:
            if custom_err_msg:
                raise NotImplementedError(custom_err_msg)
            else:
                raise NotImplementedError('Cannot call in-place function ' +
                    '{} with fp16 arguments.'.format(fn))
        else:
            return orig_fn(*args, **kwargs)
    utils.set_func_save(handle, mod, fn, wrapper)
