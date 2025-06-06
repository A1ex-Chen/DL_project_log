@BACKBONE_REGISTRY.register()
def build_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(bottom_up=bottom_up, in_features=in_features,
        out_channels=out_channels, norm=cfg.MODEL.FPN.NORM, top_block=
        LastLevelMaxPool(), fuse_type=cfg.MODEL.FPN.FUSE_TYPE)
    return backbone
