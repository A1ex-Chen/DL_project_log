def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
    dilation=1, *, norm1=None, activation1=None, norm2=None, activation2=None):
    """
        Args:
            norm1, norm2 (str or callable): normalization for the two conv layers.
            activation1, activation2 (callable(Tensor) -> Tensor): activation
                function for the two conv layers.
        """
    super().__init__()
    self.depthwise = Conv2d(in_channels, in_channels, kernel_size=
        kernel_size, padding=padding, dilation=dilation, groups=in_channels,
        bias=not norm1, norm=get_norm(norm1, in_channels), activation=
        activation1)
    self.pointwise = Conv2d(in_channels, out_channels, kernel_size=1, bias=
        not norm2, norm=get_norm(norm2, out_channels), activation=activation2)
    weight_init.c2_msra_fill(self.depthwise)
    weight_init.c2_msra_fill(self.pointwise)
