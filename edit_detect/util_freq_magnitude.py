def freq_magnitude(filtered_freqs: Union[List[torch.Tensor], torch.Tensor],
    dim: Union[Tuple[int], int], method: str='mean'):
    METHOD_SUM: str = 'sum'
    METHOD_MEAN: str = 'mean'
    if isinstance(dim, int):
        dim = [dim]
    if not isinstance(filtered_freqs, list):
        filtered_freqs = [filtered_freqs]
    if method == METHOD_SUM:
        return torch.stack([filtered_freq.sum(dim) for filtered_freq in
            filtered_freqs], dim=dim[0])
    elif method == METHOD_MEAN:
        return torch.stack([filtered_freq.mean(dim) for filtered_freq in
            filtered_freqs], dim=dim[0])
    else:
        raise ValueError(
            f'Arguement method cannot be {method}, should be {METHOD_SUM} or {METHOD_MEAN}'
            )
