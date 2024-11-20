@functools.wraps(orig_rnn)
def rnn_wrapper(*args, **kwargs):
    flat_weight = kwargs.get('flat_weight')
    if flat_weight is not None:
        assert utils.type_string(flat_weight) == 'FloatTensor'
        if compat.tensor_is_float_tensor() or compat.tensor_is_variable():
            flat_weight_fp16 = flat_weight.new().half().resize_(flat_weight
                .shape)
        else:
            flat_weight_fp16 = torch.empty_like(flat_weight, dtype=torch.
                float16)
        kwargs['flat_weight'] = flat_weight_fp16
    else:
        flat_weight_fp16 = None
    forward = orig_rnn(*args, **kwargs)

    @functools.wraps(forward)
    def fwd_wrapper(*fargs, **fkwargs):
        assert len(fargs) == 3 or len(fargs) == 4
        inputs, weights, hiddens = fargs[:3]
        assert utils.is_fp_tensor(inputs)
        assert isinstance(weights, list)
        cast_fn = utils.verbosify(utils.maybe_half, fn, verbose)
        new_args = []
        new_args.append(cast_fn(inputs))
        if flat_weight_fp16 is not None:
            fp16_weights = utils.synthesize_flattened_rnn_weights(weights,
                flat_weight_fp16, fn, verbose)
        else:
            fp16_weights = [[cast_fn(w) for w in layer] for layer in weights]
        new_args.append(fp16_weights)
        if isinstance(hiddens, tuple):
            new_args.append(tuple(cast_fn(x) for x in hiddens))
        elif utils.is_fp_tensor(hiddens):
            new_args.append(cast_fn(hiddens))
        else:
            new_args.append(hiddens)
        if len(fargs) == 4:
            new_args.append(fargs[3])
        return forward(*new_args, **fkwargs)
    return fwd_wrapper
