def build_backbone(backbone_name, **kwargs):
    if backbone_name not in BACKBONE_MAP:
        raise ValueError(
            f'Backbone {backbone_name} not supported. Supported backbones: {BACKBONE_MAP.keys()}'
            )
    backbone = BACKBONE_MAP[backbone_name](**kwargs)
    return backbone
