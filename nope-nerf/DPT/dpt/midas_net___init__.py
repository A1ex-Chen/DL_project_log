def __init__(self, path=None, features=256, non_negative=True):
    """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
    print('Loading weights: ', path)
    super(MidasNet_large, self).__init__()
    use_pretrained = False if path is None else True
    self.pretrained, self.scratch = _make_encoder(backbone='resnext101_wsl',
        features=features, use_pretrained=use_pretrained)
    self.scratch.refinenet4 = FeatureFusionBlock(features)
    self.scratch.refinenet3 = FeatureFusionBlock(features)
    self.scratch.refinenet2 = FeatureFusionBlock(features)
    self.scratch.refinenet1 = FeatureFusionBlock(features)
    self.scratch.output_conv = nn.Sequential(nn.Conv2d(features, 128,
        kernel_size=3, stride=1, padding=1), Interpolate(scale_factor=2,
        mode='bilinear'), nn.Conv2d(128, 32, kernel_size=3, stride=1,
        padding=1), nn.ReLU(True), nn.Conv2d(32, 1, kernel_size=1, stride=1,
        padding=0), nn.ReLU(True) if non_negative else nn.Identity())
    if path:
        self.load(path)
