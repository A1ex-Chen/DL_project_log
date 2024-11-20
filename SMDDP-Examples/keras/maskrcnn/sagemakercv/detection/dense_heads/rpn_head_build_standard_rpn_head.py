@HEADS.register('StandardRPNHead')
def build_standard_rpn_head(cfg):
    head_type = StandardRPNHead
    return head_type(rpn_loss_cfg=dict(min_level=cfg.MODEL.DENSE.MIN_LEVEL,
        max_level=cfg.MODEL.DENSE.MAX_LEVEL, box_loss_type=cfg.MODEL.DENSE.
        LOSS_TYPE, train_batch_size_per_gpu=cfg.INPUT.TRAIN_BATCH_SIZE //
        MPI_size(), rpn_batch_size_per_im=cfg.MODEL.DENSE.
        BATCH_SIZE_PER_IMAGE, label_smoothing=cfg.MODEL.DENSE.
        LABEL_SMOOTHING, rpn_box_loss_weight=cfg.MODEL.DENSE.LOSS_WEIGHT),
        anchor_generator_cfg=dict(min_level=cfg.MODEL.DENSE.MIN_LEVEL,
        max_level=cfg.MODEL.DENSE.MAX_LEVEL, num_scales=cfg.MODEL.DENSE.
        NUM_SCALES, aspect_ratios=cfg.MODEL.DENSE.ASPECT_RATIOS,
        anchor_scale=cfg.MODEL.DENSE.ANCHOR_SCALE, image_size=cfg.INPUT.
        IMAGE_SIZE), roi_proposal_cfg=dict(train_cfg=dict(rpn_pre_nms_topn=
        cfg.MODEL.DENSE.PRE_NMS_TOP_N_TRAIN, rpn_post_nms_topn=cfg.MODEL.
        DENSE.POST_NMS_TOP_N_TRAIN, rpn_nms_threshold=cfg.MODEL.DENSE.
        NMS_THRESH), test_cfg=dict(rpn_pre_nms_topn=cfg.MODEL.DENSE.
        PRE_NMS_TOP_N_TEST, rpn_post_nms_topn=cfg.MODEL.DENSE.
        POST_NMS_TOP_N_TEST, rpn_nms_threshold=cfg.MODEL.DENSE.NMS_THRESH),
        rpn_min_size=cfg.MODEL.DENSE.MIN_SIZE, use_custom_box_proposals_op=
        cfg.MODEL.DENSE.USE_FAST_BOX_PROPOSAL, use_batched_nms=cfg.MODEL.
        DENSE.USE_BATCHED_NMS, bbox_reg_weights=cfg.MODEL.BBOX_REG_WEIGHTS),
        num_classes=1, feat_channels=cfg.MODEL.DENSE.FEAT_CHANNELS,
        trainable=cfg.MODEL.DENSE.TRAINABLE)
