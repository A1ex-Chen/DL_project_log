def cast_if_src_dtype(tensor: torch.Tensor, src_dtype: torch.dtype,
    tgt_dtype: torch.dtype):
    updated = False
    if tensor.dtype == src_dtype:
        tensor = tensor.to(dtype=tgt_dtype)
        updated = True
    return tensor, updated
