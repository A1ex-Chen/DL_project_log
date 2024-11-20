def build_roi_head(cfg):
    return HEADS[cfg.MODEL.RCNN.ROI_HEAD](cfg)
