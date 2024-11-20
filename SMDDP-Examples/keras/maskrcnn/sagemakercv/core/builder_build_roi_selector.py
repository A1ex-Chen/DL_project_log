def build_roi_selector(cfg):
    """
    Reads standard configuration file to build
    NMS region of interest selector
    """
    roi_selector = ROI_SELECTORS['ProposeROIs']
    return roi_selector(train_cfg=dict(rpn_pre_nms_topn=cfg.MODEL.RPN.
        PRE_NMS_TOP_N_TRAIN, rpn_post_nms_topn=cfg.MODEL.RPN.
        POST_NMS_TOP_N_TRAIN, rpn_nms_threshold=cfg.MODEL.RPN.NMS_THRESH),
        test_cfg=dict(rpn_pre_nms_topn=cfg.MODEL.RPN.PRE_NMS_TOP_N_TEST,
        rpn_post_nms_topn=cfg.MODEL.RPN.POST_NMS_TOP_N_TEST,
        rpn_nms_threshold=cfg.MODEL.RPN.NMS_THRESH), rpn_min_size=cfg.MODEL
        .RPN.MIN_SIZE, use_custom_box_proposals_op=cfg.MODEL.RPN.
        USE_FAST_BOX_PROPOSAL, use_batched_nms=cfg.MODEL.RPN.
        USE_BATCHED_NMS, bbox_reg_weights=cfg.MODEL.BBOX_REG_WEIGHTS)
