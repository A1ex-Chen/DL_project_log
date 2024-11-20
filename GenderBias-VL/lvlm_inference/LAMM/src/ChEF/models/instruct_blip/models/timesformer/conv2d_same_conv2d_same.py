def conv2d_same(x, weight: torch.Tensor, bias: Optional[torch.Tensor]=None,
    stride: Tuple[int, int]=(1, 1), padding: Tuple[int, int]=(0, 0),
    dilation: Tuple[int, int]=(1, 1), groups: int=1):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
