def __init__(self, in_channels, out_channels, kernel_size, stride=1,
    padding=0, dilation=1, groups=1, deformable_groups=1, bias=True, norm=
    None, activation=None):
    """
        Modulated deformable convolution from :paper:`deformconv2`.

        Arguments are similar to :class:`Conv2D`. Extra arguments:

        Args:
            deformable_groups (int): number of groups used in deformable convolution.
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        """
    super(ModulatedDeformConv, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = _pair(kernel_size)
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
    self.deformable_groups = deformable_groups
    self.with_bias = bias
    self.norm = norm
    self.activation = activation
    self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels //
        groups, *self.kernel_size))
    if bias:
        self.bias = nn.Parameter(torch.Tensor(out_channels))
    else:
        self.bias = None
    nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
    if self.bias is not None:
        nn.init.constant_(self.bias, 0)
