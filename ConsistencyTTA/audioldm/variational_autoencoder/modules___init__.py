def __init__(self, ch_mult: list, in_channels, pretrained_model: nn.Module=
    None, reshape=False, n_channels=None, dropout=0.0, pretrained_config=None):
    super().__init__()
    if pretrained_config is None:
        assert pretrained_model is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
        self.pretrained_model = pretrained_model
    else:
        assert pretrained_config is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
        self.instantiate_pretrained(pretrained_config)
    self.do_reshape = reshape
    if n_channels is None:
        n_channels = self.pretrained_model.encoder.ch
    self.proj_norm = Normalize(in_channels, num_groups=in_channels // 2)
    self.proj = nn.Conv2d(in_channels, n_channels, kernel_size=3, stride=1,
        padding=1)
    blocks = []
    downs = []
    ch_in = n_channels
    for m in ch_mult:
        blocks.append(ResnetBlock(in_channels=ch_in, out_channels=m *
            n_channels, dropout=dropout))
        ch_in = m * n_channels
        downs.append(Downsample(ch_in, with_conv=False))
    self.model = nn.ModuleList(blocks)
    self.downsampler = nn.ModuleList(downs)
