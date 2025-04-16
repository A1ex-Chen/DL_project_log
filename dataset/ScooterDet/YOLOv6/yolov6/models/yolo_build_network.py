def build_network(config, channels, num_classes, num_layers, fuse_ab=False,
    distill_ns=False):
    depth_mul = config.model.depth_multiple
    width_mul = config.model.width_multiple
    num_repeat_backbone = config.model.backbone.num_repeats
    channels_list_backbone = config.model.backbone.out_channels
    fuse_P2 = config.model.backbone.get('fuse_P2')
    cspsppf = config.model.backbone.get('cspsppf')
    num_repeat_neck = config.model.neck.num_repeats
    channels_list_neck = config.model.neck.out_channels
    use_dfl = config.model.head.use_dfl
    reg_max = config.model.head.reg_max
    num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in 
        num_repeat_backbone + num_repeat_neck]
    channels_list = [make_divisible(i * width_mul, 8) for i in 
        channels_list_backbone + channels_list_neck]
    block = get_block(config.training_mode)
    BACKBONE = eval(config.model.backbone.type)
    NECK = eval(config.model.neck.type)
    if 'CSP' in config.model.backbone.type:
        if 'stage_block_type' in config.model.backbone:
            stage_block_type = config.model.backbone.stage_block_type
        else:
            stage_block_type = 'BepC3'
        backbone = BACKBONE(in_channels=channels, channels_list=
            channels_list, num_repeats=num_repeat, block=block, csp_e=
            config.model.backbone.csp_e, fuse_P2=fuse_P2, cspsppf=cspsppf,
            stage_block_type=stage_block_type)
        neck = NECK(channels_list=channels_list, num_repeats=num_repeat,
            block=block, csp_e=config.model.neck.csp_e, stage_block_type=
            stage_block_type)
    else:
        backbone = BACKBONE(in_channels=channels, channels_list=
            channels_list, num_repeats=num_repeat, block=block, fuse_P2=
            fuse_P2, cspsppf=cspsppf)
        neck = NECK(channels_list=channels_list, num_repeats=num_repeat,
            block=block)
    if distill_ns:
        from yolov6.models.heads.effidehead_distill_ns import Detect, build_effidehead_layer
        if num_layers != 3:
            LOGGER.error(
                'ERROR in: Distill mode not fit on n/s models with P6 head.\n')
            exit()
        head_layers = build_effidehead_layer(channels_list, 1, num_classes,
            reg_max=reg_max)
        head = Detect(num_classes, num_layers, head_layers=head_layers,
            use_dfl=use_dfl)
    elif fuse_ab:
        from yolov6.models.heads.effidehead_fuseab import Detect, build_effidehead_layer
        anchors_init = config.model.head.anchors_init
        head_layers = build_effidehead_layer(channels_list, 3, num_classes,
            reg_max=reg_max, num_layers=num_layers)
        head = Detect(num_classes, anchors_init, num_layers, head_layers=
            head_layers, use_dfl=use_dfl)
    else:
        from yolov6.models.effidehead import Detect, build_effidehead_layer
        head_layers = build_effidehead_layer(channels_list, 1, num_classes,
            reg_max=reg_max, num_layers=num_layers)
        head = Detect(num_classes, num_layers, head_layers=head_layers,
            use_dfl=use_dfl)
    return backbone, neck, head
