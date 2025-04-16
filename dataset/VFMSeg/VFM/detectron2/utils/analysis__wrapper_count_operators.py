def _wrapper_count_operators(model: nn.Module, inputs: list, mode: str, **
    kwargs) ->typing.DefaultDict[str, float]:
    supported_ops = {k: (lambda *args, **kwargs: {}) for k in _IGNORED_OPS}
    supported_ops.update(kwargs.pop('supported_ops', {}))
    kwargs['supported_ops'] = supported_ops
    assert len(inputs) == 1, 'Please use batch size=1'
    tensor_input = inputs[0]['image']
    inputs = [{'image': tensor_input}]
    old_train = model.training
    if isinstance(model, (nn.parallel.distributed.DistributedDataParallel,
        nn.DataParallel)):
        model = model.module
    wrapper = TracingAdapter(model, inputs)
    wrapper.eval()
    if mode == FLOPS_MODE:
        ret = flop_count(wrapper, (tensor_input,), **kwargs)
    elif mode == ACTIVATIONS_MODE:
        ret = activation_count(wrapper, (tensor_input,), **kwargs)
    else:
        raise NotImplementedError('Count for mode {} is not supported yet.'
            .format(mode))
    if isinstance(ret, tuple):
        ret = ret[0]
    model.train(old_train)
    return ret
