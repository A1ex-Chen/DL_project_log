def build_box_head(cfg):
    return HEADS[cfg.MODEL.FRCNN.BBOX_HEAD](cfg)
