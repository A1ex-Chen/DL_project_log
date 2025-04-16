@classmethod
def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
    in_features = cfg.MODEL.RPN.IN_FEATURES
    ret = {'in_features': in_features, 'min_box_size': cfg.MODEL.
        PROPOSAL_GENERATOR.MIN_SIZE, 'nms_thresh': cfg.MODEL.RPN.NMS_THRESH,
        'batch_size_per_image': cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
        'positive_fraction': cfg.MODEL.RPN.POSITIVE_FRACTION, 'loss_weight':
        {'loss_rpn_cls': cfg.MODEL.RPN.LOSS_WEIGHT, 'loss_rpn_loc': cfg.
        MODEL.RPN.BBOX_REG_LOSS_WEIGHT * cfg.MODEL.RPN.LOSS_WEIGHT},
        'anchor_boundary_thresh': cfg.MODEL.RPN.BOUNDARY_THRESH,
        'box2box_transform': Box2BoxTransform(weights=cfg.MODEL.RPN.
        BBOX_REG_WEIGHTS), 'box_reg_loss_type': cfg.MODEL.RPN.
        BBOX_REG_LOSS_TYPE, 'smooth_l1_beta': cfg.MODEL.RPN.SMOOTH_L1_BETA}
    ret['pre_nms_topk'
        ] = cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, cfg.MODEL.RPN.PRE_NMS_TOPK_TEST
    ret['post_nms_topk'
        ] = cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST
    ret['anchor_generator'] = build_anchor_generator(cfg, [input_shape[f] for
        f in in_features])
    ret['anchor_matcher'] = Matcher(cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL
        .RPN.IOU_LABELS, allow_low_quality_matches=True)
    ret['head'] = build_rpn_head(cfg, [input_shape[f] for f in in_features])
    return ret
