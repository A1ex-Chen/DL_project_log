def build_network(config, in_channels, num_classes):
    width_mul = config.model.width_multiple
    num_repeat_backbone = config.model.backbone.num_repeats
    out_channels_backbone = config.model.backbone.out_channels
    scale_size_backbone = config.model.backbone.scale_size
    in_channels_neck = config.model.neck.in_channels
    unified_channels_neck = config.model.neck.unified_channels
    in_channels_head = config.model.head.in_channels
    num_layers = config.model.head.num_layers
    BACKBONE = eval(config.model.backbone.type)
    NECK = eval(config.model.neck.type)
    out_channels_backbone = [make_divisible(i * width_mul) for i in
        out_channels_backbone]
    mid_channels_backbone = [make_divisible(int(i * scale_size_backbone),
        divisor=8) for i in out_channels_backbone]
    in_channels_neck = [make_divisible(i * width_mul) for i in in_channels_neck
        ]
    backbone = BACKBONE(in_channels, mid_channels_backbone,
        out_channels_backbone, num_repeat=num_repeat_backbone)
    neck = NECK(in_channels_neck, unified_channels_neck)
    head_layers = build_effidehead_layer(in_channels_head, 1, num_classes,
        num_layers)
    head = Detect(num_classes, num_layers, head_layers=head_layers)
    return backbone, neck, head
