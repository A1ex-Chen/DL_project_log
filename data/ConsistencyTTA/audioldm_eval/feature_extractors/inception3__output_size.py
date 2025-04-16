def _output_size(dim):
    _check_size_scale_factor(dim)
    if size is not None:
        if is_tracing:
            return [torch.tensor(i) for i in size]
        else:
            return size
    scale_factors = _ntuple(dim)(scale_factor)
    if is_tracing:
        return [torch.floor((input.size(i + 2).float() * torch.tensor(
            scale_factors[i], dtype=torch.float32)).float()) for i in range
            (dim)]
    else:
        return [int(math.floor(float(input.size(i + 2)) * scale_factors[i])
            ) for i in range(dim)]
