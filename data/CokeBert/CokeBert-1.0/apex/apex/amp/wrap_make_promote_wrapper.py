def make_promote_wrapper(orig_fn, cast_fn, handle=None):

    @functools.wraps(orig_fn)
    def wrapper(*args, **kwargs):
        if not _amp_state.handle.is_active():
            return orig_fn(*args, **kwargs)
        types = utils.collect_fp_tensor_types(args, kwargs)
        if len(types) <= 1:
            return orig_fn(*args, **kwargs)
        elif len(types) == 2 and types == set(['HalfTensor', 'FloatTensor']):
            new_args = utils.casted_args(cast_fn, args, kwargs)
            return orig_fn(*new_args, **kwargs)
        else:
            raise NotImplementedError('Do not know how to handle ' +
                'these types to promote: {}'.format(types))
    return wrapper
