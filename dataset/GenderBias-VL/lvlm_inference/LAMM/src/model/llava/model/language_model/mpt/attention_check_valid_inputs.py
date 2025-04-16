def check_valid_inputs(*tensors, valid_dtypes=[torch.float16, torch.bfloat16]):
    for tensor in tensors:
        if tensor.dtype not in valid_dtypes:
            raise TypeError(
                f'tensor.dtype={tensor.dtype!r} must be in valid_dtypes={valid_dtypes!r}.'
                )
        if not tensor.is_cuda:
            raise TypeError(
                f'Inputs must be cuda tensors (tensor.is_cuda={tensor.is_cuda!r}).'
                )
