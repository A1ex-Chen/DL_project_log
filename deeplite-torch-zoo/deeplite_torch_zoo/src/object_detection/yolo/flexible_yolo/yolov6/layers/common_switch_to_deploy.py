def switch_to_deploy(self):
    if hasattr(self, 'rbr_reparam'):
        return
    kernel, bias = self.get_equivalent_kernel_bias()
    self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.
        in_channels, out_channels=self.rbr_dense.conv.out_channels,
        kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.
        conv.stride, padding=self.rbr_dense.conv.padding, dilation=self.
        rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
    self.rbr_reparam.weight.data = kernel
    self.rbr_reparam.bias.data = bias
    for para in self.parameters():
        para.detach_()
    self.__delattr__('rbr_dense')
    self.__delattr__('rbr_1x1')
    if hasattr(self, 'rbr_identity'):
        self.__delattr__('rbr_identity')
    if hasattr(self, 'rbr_avg'):
        self.__delattr__('rbr_avg')
    if hasattr(self, 'id_tensor'):
        self.__delattr__('id_tensor')
    self.deploy = True
