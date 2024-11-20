def new_rnn_cast(fn, handle, verbose=False):
    if utils.has_func(torch.nn.modules.rnn._rnn_impls, fn):
        mod = torch.nn.modules.rnn._rnn_impls
    else:
        mod = torch.nn.modules.rnn._VF
        assert isinstance(mod, rnn_compat.VariableFunctionsShim)
        fn = fn.lower()
    orig_fn = utils.get_func(mod, fn)
    cast_fn = utils.verbosify(utils.maybe_half, fn, verbose)

    @functools.wraps(orig_fn)
    def wrapper(*args, **kwargs):
        assert len(args) == 9
        assert len(kwargs) == 0
        if not _amp_state.handle.is_active():
            return orig_fn(*args, **kwargs)
        if isinstance(args[6], bool):
            params_idx = 2
        else:
            params_idx = 3
        new_args = []
        for i, arg in enumerate(args):
            if i == params_idx:
                num_params = sum([x.numel() for x in arg])
                fp16_weight_buf = args[0].new_empty((num_params,), dtype=
                    torch.half)
                casted_weights = utils.new_synthesize_flattened_rnn_weights(arg
                    , fp16_weight_buf, fn, verbose)
                new_args.append(casted_weights)
            elif utils.is_fp_tensor(arg):
                new_args.append(cast_fn(arg))
            else:
                new_args.append(arg)
        return orig_fn(*new_args)
    utils.set_func_save(handle, mod, fn, wrapper)
