def build_network_lite(config, channels=3):
    width_mul = config.model.width_multiple
    num_repeat_backbone = config.model.backbone.num_repeats
    out_channels_backbone = config.model.backbone.out_channels
    scale_size_backbone = config.model.backbone.scale_size
    in_channels_neck = config.model.neck.in_channels
    unified_channels_neck = config.model.neck.unified_channels
    in_channels_head = config.model.head.in_channels
    BACKBONE = eval(config.model.backbone.type)
    NECK = eval(config.model.neck.type)
    out_channels_backbone = [make_divisible(i * width_mul, divisor=16) for
        i in out_channels_backbone]
    mid_channels_backbone = [make_divisible(int(i * scale_size_backbone),
        divisor=8) for i in out_channels_backbone]
    in_channels_neck = [make_divisible(i * width_mul, divisor=16) for i in
        in_channels_neck]
    backbone = BACKBONE(channels, mid_channels_backbone,
        out_channels_backbone, num_repeat=num_repeat_backbone)
    neck = NECK(in_channels_neck, unified_channels_neck)
    return backbone, neck, in_channels_head
