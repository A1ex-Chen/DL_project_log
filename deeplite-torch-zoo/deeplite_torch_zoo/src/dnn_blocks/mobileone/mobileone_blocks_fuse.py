def fuse(self):
    """Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
    LOGGER.info(f'fusing a MobileOne block')
    if self.inference_mode:
        return
    kernel, bias = self._get_kernel_bias()
    self.reparam_conv = nn.Conv2d(in_channels=self.rbr_conv[0].conv.
        in_channels, out_channels=self.rbr_conv[0].conv.out_channels,
        kernel_size=self.rbr_conv[0].conv.kernel_size, stride=self.rbr_conv
        [0].conv.stride, padding=self.rbr_conv[0].conv.padding, dilation=
        self.rbr_conv[0].conv.dilation, groups=self.rbr_conv[0].conv.groups,
        bias=True)
    self.reparam_conv.weight.data = kernel
    self.reparam_conv.bias.data = bias
    for para in self.parameters():
        para.detach_()
    self.__delattr__('rbr_conv')
    self.__delattr__('rbr_scale')
    if hasattr(self, 'rbr_skip'):
        self.__delattr__('rbr_skip')
    self.inference_mode = True
