def _make_stem(self, stem_width):
    return nn.Sequential(OrderedDict([('conv', self.builder.conv3x3(3,
        stem_width, stride=2)), ('bn', self.builder.batchnorm(stem_width)),
        ('activation', self.builder.activation())]))
