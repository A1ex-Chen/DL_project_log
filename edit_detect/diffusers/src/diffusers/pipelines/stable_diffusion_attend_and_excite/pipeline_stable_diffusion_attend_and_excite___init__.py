def __init__(self, channels: int=1, kernel_size: int=3, sigma: float=0.5,
    dim: int=2):
    super().__init__()
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * dim
    if isinstance(sigma, float):
        sigma = [sigma] * dim
    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for
        size in kernel_size])
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid -
            mean) / (2 * std)) ** 2)
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *([1] * (kernel.dim() - 1)))
    self.register_buffer('weight', kernel)
    self.groups = channels
    if dim == 1:
        self.conv = F.conv1d
    elif dim == 2:
        self.conv = F.conv2d
    elif dim == 3:
        self.conv = F.conv3d
    else:
        raise RuntimeError(
            'Only 1, 2 and 3 dimensions are supported. Received {}.'.format
            (dim))
