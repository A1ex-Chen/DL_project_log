def switch_to_deploy(self):
    if hasattr(self, 'rbr_reparam'):
        return
    print(f'RepConv_OREPA.switch_to_deploy')
    kernel, bias = self.get_equivalent_kernel_bias()
    self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.in_channels,
        out_channels=self.rbr_dense.out_channels, kernel_size=self.
        rbr_dense.kernel_size, stride=self.rbr_dense.stride, padding=self.
        rbr_dense.padding, dilation=self.rbr_dense.dilation, groups=self.
        rbr_dense.groups, bias=True)
    self.rbr_reparam.weight.data = kernel
    self.rbr_reparam.bias.data = bias
    for para in self.parameters():
        para.detach_()
    self.__delattr__('rbr_dense')
    self.__delattr__('rbr_1x1')
    if hasattr(self, 'rbr_identity'):
        self.__delattr__('rbr_identity')
