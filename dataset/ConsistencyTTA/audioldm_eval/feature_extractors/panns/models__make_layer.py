def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if dilate:
        self.dilation *= stride
        stride = 1
    if stride != 1 or self.inplanes != planes * block.expansion:
        if stride == 1:
            downsample = nn.Sequential(_resnet_conv1x1_wav1d(self.inplanes,
                planes * block.expansion), norm_layer(planes * block.expansion)
                )
            init_layer(downsample[0])
            init_bn(downsample[1])
        else:
            downsample = nn.Sequential(nn.AvgPool1d(kernel_size=stride),
                _resnet_conv1x1_wav1d(self.inplanes, planes * block.
                expansion), norm_layer(planes * block.expansion))
            init_layer(downsample[1])
            init_bn(downsample[2])
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, self.
        groups, self.base_width, previous_dilation, norm_layer))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(self.inplanes, planes, groups=self.groups,
            base_width=self.base_width, dilation=self.dilation, norm_layer=
            norm_layer))
    return nn.Sequential(*layers)
