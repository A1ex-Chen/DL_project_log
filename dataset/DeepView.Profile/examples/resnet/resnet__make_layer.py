def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if dilate:
        self.dilation *= stride
        stride = 1
    if stride != 1 or self.inplanes != planes * block.expansion:
        downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.
            expansion, stride), norm_layer(planes * block.expansion))
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, self.
        groups, self.base_width, previous_dilation, norm_layer))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(self.inplanes, planes, groups=self.groups,
            base_width=self.base_width, dilation=self.dilation, norm_layer=
            norm_layer))
    return nn.Sequential(*layers)
