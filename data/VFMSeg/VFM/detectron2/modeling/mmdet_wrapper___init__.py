def __init__(self, detector: Union[nn.Module, Mapping], *,
    size_divisibility=32, pixel_mean: Tuple[float], pixel_std: Tuple[float]):
    """
        Args:
            detector: a mmdet detector, or a mmdet config dict that defines a detector.
            size_divisibility: pad input images to multiple of this number
            pixel_mean: per-channel mean to normalize input image
            pixel_std: per-channel stddev to normalize input image
        """
    super().__init__()
    if isinstance(detector, Mapping):
        from mmdet.models import build_detector
        detector = build_detector(_to_container(detector))
    self.detector = detector
    self.size_divisibility = size_divisibility
    self.register_buffer('pixel_mean', torch.tensor(pixel_mean).view(-1, 1,
        1), False)
    self.register_buffer('pixel_std', torch.tensor(pixel_std).view(-1, 1, 1
        ), False)
    assert self.pixel_mean.shape == self.pixel_std.shape, f'{self.pixel_mean} and {self.pixel_std} have different shapes!'
