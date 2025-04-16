@classmethod
def _build_res5_block(cls, cfg):
    stage_channel_factor = 2 ** 3
    num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group * stage_channel_factor
    out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
    stride_in_1x1 = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    norm = cfg.MODEL.RESNETS.NORM
    assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1
        ], 'Deformable conv is not yet supported in res5 head.'
    blocks = ResNet.make_stage(BottleneckBlock, 3, stride_per_block=[2, 1, 
        1], in_channels=out_channels // 2, bottleneck_channels=
        bottleneck_channels, out_channels=out_channels, num_groups=
        num_groups, norm=norm, stride_in_1x1=stride_in_1x1)
    return nn.Sequential(*blocks), out_channels
