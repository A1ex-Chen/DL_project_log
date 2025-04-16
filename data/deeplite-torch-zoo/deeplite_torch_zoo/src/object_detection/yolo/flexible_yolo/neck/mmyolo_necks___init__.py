def __init__(self, ch=(256, 512, 1024), channel_outs=(256, 512, 1024),
    version='l', width_factor=None, depth_factor=None, act='silu',
    num_csplayer=1, num_blocks_per_layer=3, drop_block_cfg=None, use_spp=True):
    if width_factor is None:
        width_factor = YOLO_SCALING_GAINS[version.lower()]['gw']
    if depth_factor is None:
        depth_factor = YOLO_SCALING_GAINS[version.lower()]['gd']
    super().__init__(in_channels=ch, out_channels=channel_outs,
        deepen_factor=depth_factor, widen_factor=width_factor, act_cfg=
        NECK_ACT_TYPE_MAP[act], num_csplayer=num_csplayer,
        num_blocks_per_layer=num_blocks_per_layer, block_cfg=dict(type=
        'PPYOLOEBasicBlock', shortcut=False, use_alpha=False), norm_cfg=
        dict(type='BN', momentum=0.1, eps=1e-05), drop_block_cfg=
        drop_block_cfg, use_spp=use_spp)
    self.out_shape = channel_outs
