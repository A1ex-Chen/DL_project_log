def build_mask_head(cfg):
    return HEADS[cfg.MODEL.MRCNN.MASK_HEAD](cfg)
