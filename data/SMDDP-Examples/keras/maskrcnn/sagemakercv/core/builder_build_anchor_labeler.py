def build_anchor_labeler(cfg, anchor_generator):
    """
    Reads standard configuration file to build
    anchor labeler
    """
    anchor_type = ANCHORS['AnchorLabeler']
    return anchor_type(anchors=anchor_generator, num_classes=cfg.INPUT.
        NUM_CLASSES, match_threshold=cfg.MODEL.RPN.POSITIVE_OVERLAP,
        unmatched_threshold=cfg.MODEL.RPN.NEGATIVE_OVERLAP,
        rpn_batch_size_per_im=cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
        rpn_fg_fraction=cfg.MODEL.RPN.FG_FRACTION)
