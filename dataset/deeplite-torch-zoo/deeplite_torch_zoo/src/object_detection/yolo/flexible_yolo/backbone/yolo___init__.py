def __init__(self, version='s', num_blocks=(3, 6, 6, 3),
    bottleneck_block_cls=None, act='silu', channels_out=(64, 128, 128, 256,
    256, 512, 512, 1024, 1024, 1024), depth_factor=None, width_factor=None):
    self.version = version
    if depth_factor is None:
        self.gd = YOLO_SCALING_GAINS[self.version.lower()]['gd']
    else:
        self.gd = depth_factor
    if bottleneck_block_cls is None:
        bottleneck_block_cls = [partial(YOLOC2f, shortcut=True, act=act, n=
            self.get_depth(n)) for n in num_blocks]
    super().__init__(version=version, num_blocks=num_blocks,
        bottleneck_block_cls=bottleneck_block_cls, act=act, channels_out=
        channels_out, depth_factor=depth_factor, width_factor=width_factor)
    self.C1 = Conv(3, self.channels_out[0], 3, 2)
