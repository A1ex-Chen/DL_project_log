def build_anchor_generator(cfg):
    """
    Reads standard configuration file to build
    anchor generator
    """
    anchor_type = ANCHORS['AnchorGenerator']
    return anchor_type(min_level=cfg.MODEL.RPN.MIN_LEVEL, max_level=cfg.
        MODEL.RPN.MAX_LEVEL, num_scales=cfg.MODEL.RPN.NUM_SCALES,
        aspect_ratios=cfg.MODEL.RPN.ASPECT_RATIOS, anchor_scale=cfg.MODEL.
        RPN.ANCHOR_SCALE, image_size=cfg.INPUT.IMAGE_SIZE)
