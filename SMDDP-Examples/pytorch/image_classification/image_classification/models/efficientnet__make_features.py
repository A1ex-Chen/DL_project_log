def _make_features(self, in_channels, num_features):
    return nn.Sequential(OrderedDict([('conv', self.builder.conv1x1(
        in_channels, num_features)), ('bn', self.builder.batchnorm(
        num_features)), ('activation', self.builder.activation())]))
