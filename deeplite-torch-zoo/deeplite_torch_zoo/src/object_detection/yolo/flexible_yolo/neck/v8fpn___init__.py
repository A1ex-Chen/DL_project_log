def __init__(self, ch=(256, 512, 1024), channel_outs=(512, 256), version=
    's', default_gd=0.33, default_gw=0.5, bottleneck_block_cls=None,
    bottleneck_depth=(3, 3), act='silu', channel_divisor=8):
    super().__init__()
    self.C3_size = ch[0]
    self.C4_size = ch[1]
    self.C5_size = ch[2]
    self.channels_outs = channel_outs
    self.channel_divisor = channel_divisor
    self.version = version
    self.gd = default_gd
    self.gw = default_gw
    if self.version is not None and self.version.lower() in YOLO_SCALING_GAINS:
        self.gd = YOLO_SCALING_GAINS[self.version.lower()]['gd']
        self.gw = YOLO_SCALING_GAINS[self.version.lower()]['gw']
    self.re_channels_out()
    self.concat = Concat()
    self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
    if bottleneck_block_cls is None:
        bottleneck_block_cls = [partial(YOLOC2f, shortcut=False, n=self.
            get_depth(n), act=act) for n in bottleneck_depth]
    self.conv1 = bottleneck_block_cls[0](self.C5_size + self.C4_size, self.
        channels_outs[0])
    self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
    self.P3 = bottleneck_block_cls[1](self.channels_outs[0] + self.C3_size,
        self.channels_outs[1])
    self.out_shape = self.channels_outs[1], self.channels_outs[0], self.C5_size
    LOGGER.info(
        f'FPN input channel sizes: C3 {self.C3_size}, C4 {self.C4_size}, C5 {self.C5_size}'
        )
    LOGGER.info(
        f'FPN output channel sizes: P3 {self.channels_outs[1]}, P4 {self.channels_outs[0]}, P5 {self.C5_size}'
        )
