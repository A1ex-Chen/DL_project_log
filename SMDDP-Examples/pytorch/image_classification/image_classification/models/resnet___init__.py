def __init__(self, arch: Arch, num_classes: int=1000, last_bn_0_init: bool=
    False, conv_init: str='fan_in', trt: bool=False):
    super(ResNet, self).__init__()
    self.arch = arch
    self.builder = LayerBuilder(LayerBuilder.Config(activation=arch.
        activation, conv_init=conv_init))
    self.last_bn_0_init = last_bn_0_init
    self.conv1 = self.builder.conv7x7(3, arch.stem_width, stride=2)
    self.bn1 = self.builder.batchnorm(arch.stem_width)
    self.relu = self.builder.activation()
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    inplanes = arch.stem_width
    assert len(arch.widths) == len(arch.layers)
    self.num_layers = len(arch.widths)
    for i, (w, l) in enumerate(zip(arch.widths, arch.layers)):
        layer, inplanes = self._make_layer(arch.block, arch.expansion,
            inplanes, w, l, cardinality=arch.cardinality, stride=1 if i == 
            0 else 2, trt=trt)
        setattr(self, f'layer{i + 1}', layer)
    self.avgpool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Linear(arch.widths[-1] * arch.expansion, num_classes)
