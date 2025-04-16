def build_backbone(cfg):
    backbone_type = BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY]
    return backbone_type(sub_type=cfg.MODEL.BACKBONE.CONV_BODY, data_format
        =cfg.MODEL.BACKBONE.DATA_FORMAT, trainable=cfg.MODEL.BACKBONE.
        TRAINABLE, finetune_bn=cfg.MODEL.BACKBONE.FINETUNE_BN, norm_type=
        cfg.MODEL.BACKBONE.NORM_TYPE)
