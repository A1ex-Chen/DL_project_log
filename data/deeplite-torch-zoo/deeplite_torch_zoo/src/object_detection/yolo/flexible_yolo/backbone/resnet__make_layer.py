def _make_layer(self, block, planes, blocks, stride=1, cbam=False, dcn=None):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
        downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.
            expansion, kernel_size=1, stride=stride, bias=False), nn.
            BatchNorm2d(planes * block.expansion))
    layers = [block(self.inplanes, planes, stride, downsample, cbam=cbam,
        dcn=dcn)]
    self.inplanes = planes * block.expansion
    self.out_channels.append(self.inplanes)
    for i in range(1, blocks):
        layers.append(block(self.inplanes, planes, cbam=cbam, dcn=dcn))
    return nn.Sequential(*layers)
