def __init__(self, ch=(256, 256, 512), channel_outs=(256, 512, 512, 1024),
    version='s', default_gd=0.33, default_gw=0.5, bottleneck_block_cls=None,
    bottleneck_depth=(3, 3), act='silu', channel_divisor=8):
    super().__init__()
    self.version = str(version)
    self.channels_outs = channel_outs
    self.channel_divisor = channel_divisor
    self.gd = default_gd
    self.gw = default_gw
    if self.version is not None and self.version.lower() in YOLO_SCALING_GAINS:
        self.gd = YOLO_SCALING_GAINS[self.version.lower()]['gd']
        self.gw = YOLO_SCALING_GAINS[self.version.lower()]['gw']
    self.re_channels_out()
    self.P3_size = ch[0]
    self.P4_size = ch[1]
    self.P5_size = ch[2]
    if bottleneck_block_cls is None:
        bottleneck_block_cls = [partial(YOLOC3, shortcut=False, n=self.
            get_depth(n), act=act) for n in bottleneck_depth]
    first_stride = 2
    self.convP3 = Conv(self.P3_size, self.channels_outs[0], 3, first_stride,
        act=act)
    self.P4 = bottleneck_block_cls[0](self.channels_outs[0] + self.P4_size,
        self.channels_outs[1])
    second_stride = 2
    self.convP4 = Conv(self.channels_outs[1], self.channels_outs[2], 3,
        second_stride, act=act)
    self.P5 = bottleneck_block_cls[1](self.channels_outs[2] + self.P5_size,
        self.channels_outs[3])
    self.concat = Concat()
    self.out_shape = [self.P3_size, self.channels_outs[1], self.
        channels_outs[3]]
    LOGGER.info('PAN input channel size: P3 {}, P4 {}, P5 {}'.format(self.
        P3_size, self.P4_size, self.P5_size))
    LOGGER.info('PAN output channel size: PP3 {}, PP4 {}, PP5 {}'.format(
        self.P3_size, self.channels_outs[1], self.channels_outs[3]))
