def detach_variable(inputs):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            'Only tuple of tensors is supported. Got Unsupported input type: ',
            type(inputs).__name__)
