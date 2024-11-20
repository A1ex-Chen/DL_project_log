def _conv_bn(self, kernel_size: int, padding: int) ->nn.Sequential:
    """Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
    mod_list = nn.Sequential()
    mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,
        out_channels=self.out_channels, kernel_size=kernel_size, stride=
        self.stride, padding=padding, groups=self.groups, bias=False))
    mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
    return mod_list
