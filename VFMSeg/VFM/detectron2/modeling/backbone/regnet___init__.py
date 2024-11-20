def __init__(self, *, stem_class, stem_width, block_class, depth, w_a, w_0,
    w_m, group_width, stride=2, bottleneck_ratio=1.0, se_ratio=0.0,
    activation_class=None, freeze_at=0, norm='BN', out_features=None):
    """
        Build a RegNet from the parameterization described in :paper:`dds` Section 3.3.

        Args:
            See :class:`AnyNet` for arguments that are not listed here.
            depth (int): Total number of blocks in the RegNet.
            w_a (float): Factor by which block width would increase prior to quantizing block widths
                by stage. See :paper:`dds` Section 3.3.
            w_0 (int): Initial block width. See :paper:`dds` Section 3.3.
            w_m (float): Parameter controlling block width quantization.
                See :paper:`dds` Section 3.3.
            group_width (int): Number of channels per group in group convolution, if the block uses
                group convolution.
            bottleneck_ratio (float): The ratio of the number of bottleneck channels to the number
                of block input channels (or, equivalently, output channels), if the block uses a
                bottleneck.
            stride (int): The stride that each network stage applies to its input.
        """
    ws, ds = generate_regnet_parameters(w_a, w_0, w_m, depth)[0:2]
    ss = [stride for _ in ws]
    bs = [bottleneck_ratio for _ in ws]
    gs = [group_width for _ in ws]
    ws, bs, gs = adjust_block_compatibility(ws, bs, gs)

    def default_activation_class():
        return nn.ReLU(inplace=True)
    super().__init__(stem_class=stem_class, stem_width=stem_width,
        block_class=block_class, depths=ds, widths=ws, strides=ss,
        group_widths=gs, bottleneck_ratios=bs, se_ratio=se_ratio,
        activation_class=default_activation_class if activation_class is
        None else activation_class, freeze_at=freeze_at, norm=norm,
        out_features=out_features)
