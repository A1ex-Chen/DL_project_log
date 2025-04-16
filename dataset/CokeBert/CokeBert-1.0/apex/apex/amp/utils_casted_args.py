def casted_args(cast_fn, args, kwargs):
    new_args = []
    for x in args:
        if is_fp_tensor(x):
            new_args.append(cast_fn(x))
        else:
            new_args.append(x)
    for k in kwargs:
        val = kwargs[k]
        if is_fp_tensor(val):
            kwargs[k] = cast_fn(val)
    return new_args
