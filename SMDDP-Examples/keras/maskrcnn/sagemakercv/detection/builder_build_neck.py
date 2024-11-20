def build_neck(cfg):
    return NECKS[cfg.MODEL.BACKBONE.NECK](cfg)
