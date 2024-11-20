def fft_nd_to_1d(x: torch.Tensor, dim: Union[Tuple[int], int]=None, nd: int
    =None):
    if dim is not None and nd is not None:
        raise ValueError(
            f'Arguements dim and nd can not be used at the same time.')
    if dim is not None:
        x_fft = freq_magnitude(filtered_freqs=filtered_by_freq_all(x=x, dim
            =dim), dim=dim, method='mean')
    elif nd is not None:
        dim = list(range(len(x.shape)))[-nd:]
        x_fft = freq_magnitude(filtered_freqs=filtered_by_freq_all(x=x, dim
            =dim), dim=dim, method='mean')
    else:
        x_fft = torch.fft.fftn(x)
    return x_fft
