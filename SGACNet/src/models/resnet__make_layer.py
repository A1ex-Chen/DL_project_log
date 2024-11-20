def _make_layer(self, block, planes, blocks, stride=1, dilate=1,
    replace_stride_with_dilation=False):
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if replace_stride_with_dilation:
        self.dilation *= stride
        stride = 1
    if dilate > 1:
        self.dilation = dilate
        dilate_first_block = dilate
    else:
        dilate_first_block = previous_dilation
    if stride != 1 or self.inplanes != planes * block.expansion:
        downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.
            expansion, stride), norm_layer(planes * block.expansion))
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, self.
        groups, self.base_width, dilate_first_block, norm_layer, activation
        =self.act))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(self.inplanes, planes, groups=self.groups,
            base_width=self.base_width, dilation=self.dilation, norm_layer=
            norm_layer, activation=self.act))
    return nn.Sequential(*layers)
