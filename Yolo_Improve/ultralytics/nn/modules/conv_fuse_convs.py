def fuse_convs(self):
    """Combines two convolution layers into a single layer and removes unused attributes from the class."""
    if hasattr(self, 'conv'):
        return
    kernel, bias = self.get_equivalent_kernel_bias()
    self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
        out_channels=self.conv1.conv.out_channels, kernel_size=self.conv1.
        conv.kernel_size, stride=self.conv1.conv.stride, padding=self.conv1
        .conv.padding, dilation=self.conv1.conv.dilation, groups=self.conv1
        .conv.groups, bias=True).requires_grad_(False)
    self.conv.weight.data = kernel
    self.conv.bias.data = bias
    for para in self.parameters():
        para.detach_()
    self.__delattr__('conv1')
    self.__delattr__('conv2')
    if hasattr(self, 'nm'):
        self.__delattr__('nm')
    if hasattr(self, 'bn'):
        self.__delattr__('bn')
    if hasattr(self, 'id_tensor'):
        self.__delattr__('id_tensor')
