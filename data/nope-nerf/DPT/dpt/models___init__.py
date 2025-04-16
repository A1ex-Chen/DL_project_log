def __init__(self, path=None, non_negative=True, scale=1.0, shift=0.0,
    invert=True, freeze=True, **kwargs):
    features = kwargs['features'] if 'features' in kwargs else 256
    self.scale = scale
    self.shift = shift
    self.invert = invert
    head = nn.Sequential(nn.Conv2d(features, features // 2, kernel_size=3,
        stride=1, padding=1), Interpolate(scale_factor=2, mode='bilinear',
        align_corners=True), nn.Conv2d(features // 2, 32, kernel_size=3,
        stride=1, padding=1), nn.ReLU(True), nn.Conv2d(32, 1, kernel_size=1,
        stride=1, padding=0), nn.ReLU(True) if non_negative else nn.
        Identity(), nn.Identity())
    super().__init__(head, freeze=freeze, **kwargs)
    if path is not None:
        self.load(path)
