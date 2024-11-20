def build_dense_head(cfg):
    return HEADS[cfg.MODEL.DENSE.RPN_HEAD](cfg)
