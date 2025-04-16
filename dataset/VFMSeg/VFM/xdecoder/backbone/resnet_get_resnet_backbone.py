@register_backbone
def get_resnet_backbone(cfg):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    res_cfg = cfg['MODEL']['BACKBONE']['RESNETS']
    norm = res_cfg['NORM']
    stem = BasicStem(in_channels=res_cfg['STEM_IN_CHANNELS'], out_channels=
        res_cfg['STEM_OUT_CHANNELS'], norm=norm)
    freeze_at = res_cfg['FREEZE_AT']
    out_features = res_cfg['OUT_FEATURES']
    depth = res_cfg['DEPTH']
    num_groups = res_cfg['NUM_GROUPS']
    width_per_group = res_cfg['WIDTH_PER_GROUP']
    bottleneck_channels = num_groups * width_per_group
    in_channels = res_cfg['STEM_OUT_CHANNELS']
    out_channels = res_cfg['RES2_OUT_CHANNELS']
    stride_in_1x1 = res_cfg['STRIDE_IN_1X1']
    res5_dilation = res_cfg['RES5_DILATION']
    deform_on_per_stage = res_cfg['DEFORM_ON_PER_STAGE']
    deform_modulated = res_cfg['DEFORM_MODULATED']
    deform_num_groups = res_cfg['DEFORM_NUM_GROUPS']
    assert res5_dilation in {1, 2}, 'res5_dilation cannot be {}.'.format(
        res5_dilation)
    num_blocks_per_stage = {(18): [2, 2, 2, 2], (34): [3, 4, 6, 3], (50): [
        3, 4, 6, 3], (101): [3, 4, 23, 3], (152): [3, 8, 36, 3]}[depth]
    if depth in [18, 34]:
        assert out_channels == 64, 'Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34'
        assert not any(deform_on_per_stage
            ), 'MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34'
        assert res5_dilation == 1, 'Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34'
        assert num_groups == 1, 'Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34'
    stages = []
    for idx, stage_idx in enumerate(range(2, 6)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or stage_idx == 5 and dilation == 2 else 2
        stage_kargs = {'num_blocks': num_blocks_per_stage[idx],
            'stride_per_block': [first_stride] + [1] * (
            num_blocks_per_stage[idx] - 1), 'in_channels': in_channels,
            'out_channels': out_channels, 'norm': norm}
        if depth in [18, 34]:
            stage_kargs['block_class'] = BasicBlock
        else:
            stage_kargs['bottleneck_channels'] = bottleneck_channels
            stage_kargs['stride_in_1x1'] = stride_in_1x1
            stage_kargs['dilation'] = dilation
            stage_kargs['num_groups'] = num_groups
            if deform_on_per_stage[idx]:
                stage_kargs['block_class'] = DeformBottleneckBlock
                stage_kargs['deform_modulated'] = deform_modulated
                stage_kargs['deform_num_groups'] = deform_num_groups
            else:
                stage_kargs['block_class'] = BottleneckBlock
        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    backbone = ResNet(stem, stages, out_features=out_features, freeze_at=
        freeze_at)
    if cfg['MODEL']['BACKBONE']['LOAD_PRETRAINED'] is True:
        filename = cfg['MODEL']['BACKBONE']['PRETRAINED']
        with PathManager.open(filename, 'rb') as f:
            ckpt = pickle.load(f, encoding='latin1')['model']
        _convert_ndarray_to_tensor(ckpt)
        ckpt.pop('stem.fc.weight')
        ckpt.pop('stem.fc.bias')
        backbone.load_state_dict(ckpt)
    return backbone
