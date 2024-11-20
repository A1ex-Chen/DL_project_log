def fft_nd(x: torch.Tensor, dim: Union[Tuple[int], int]=None, nd: int=None):
    if dim is not None and nd is not None:
        raise ValueError(
            f'Arguements dim and nd can not be used at the same time.')
    if dim is not None:
        x_fft = torch.fft.fftn(x, dim=dim)
    elif nd is not None:
        x_fft = torch.fft.fftn(x, dim=x.shape[-nd:])
    else:
        x_fft = torch.fft.fftn(x)
    return torch.fft.fftshift(x_fft).abs()
