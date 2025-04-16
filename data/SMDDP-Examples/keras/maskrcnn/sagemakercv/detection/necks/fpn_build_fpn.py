@NECKS.register('FPN')
def build_fpn(cfg):
    neck = FPN
    return neck(min_level=cfg.MODEL.DENSE.MIN_LEVEL, max_level=cfg.MODEL.
        DENSE.MAX_LEVEL, filters=cfg.MODEL.DENSE.FEAT_CHANNELS, trainable=
        cfg.MODEL.BACKBONE.TRAINABLE)
