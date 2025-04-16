def __init__(self, block_out_channels: int, in_channels: int, use_bias:
    bool, ln_elementwise_affine: bool, layer_norm_eps: float, codebook_size:
    int):
    super().__init__()
    self.conv1 = nn.Conv2d(block_out_channels, in_channels, kernel_size=1,
        bias=use_bias)
    self.layer_norm = RMSNorm(in_channels, layer_norm_eps,
        ln_elementwise_affine)
    self.conv2 = nn.Conv2d(in_channels, codebook_size, kernel_size=1, bias=
        use_bias)
