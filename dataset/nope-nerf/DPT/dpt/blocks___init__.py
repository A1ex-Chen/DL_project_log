def __init__(self, features, activation, deconv=False, bn=False, expand=
    False, align_corners=True):
    """Init.

        Args:
            features (int): number of features
        """
    super(FeatureFusionBlock_custom, self).__init__()
    self.deconv = deconv
    self.align_corners = align_corners
    self.groups = 1
    self.expand = expand
    out_features = features
    if self.expand == True:
        out_features = features // 2
    self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride
        =1, padding=0, bias=True, groups=1)
    self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
    self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
    self.skip_add = nn.quantized.FloatFunctional()
