def _make_layer(self, block, expansion, inplanes, planes, blocks, stride=1,
    cardinality=1, trt=False):
    downsample = None
    if stride != 1 or inplanes != planes * expansion:
        dconv = self.builder.conv1x1(inplanes, planes * expansion, stride=
            stride)
        dbn = self.builder.batchnorm(planes * expansion)
        if dbn is not None:
            downsample = nn.Sequential(dconv, dbn)
        else:
            downsample = dconv
    layers = []
    for i in range(blocks):
        layers.append(block(self.builder, inplanes, planes, expansion,
            stride=stride if i == 0 else 1, cardinality=cardinality,
            downsample=downsample if i == 0 else None, fused_se=True,
            last_bn_0_init=self.last_bn_0_init, trt=trt))
        inplanes = planes * expansion
    return nn.Sequential(*layers), inplanes
