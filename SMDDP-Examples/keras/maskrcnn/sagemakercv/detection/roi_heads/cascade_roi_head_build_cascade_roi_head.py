@HEADS.register('CascadeRoIHead')
def build_cascade_roi_head(cfg):
    roi_head = CascadeRoIHead
    assert len(cfg.MODEL.RCNN.CASCADE.BBOX_REG_WEIGHTS) == len(cfg.MODEL.
        RCNN.CASCADE.THRESHOLDS) == len(cfg.MODEL.RCNN.CASCADE.STAGE_WEIGHTS)
    num_stages = len(cfg.MODEL.RCNN.CASCADE.THRESHOLDS)
    bbox_heads = build_box_head(cfg)
    bbox_roi_extractor = GenericRoIExtractor(cfg.MODEL.FRCNN.ROI_SIZE, cfg.
        MODEL.FRCNN.GPU_INFERENCE)
    bbox_samplers = [RandomSampler(batch_size_per_im=cfg.MODEL.RCNN.
        BATCH_SIZE_PER_IMAGE, fg_fraction=cfg.MODEL.RCNN.FG_FRACTION,
        fg_thresh=stage_tresh, bg_thresh_hi=stage_tresh, bg_thresh_lo=cfg.
        MODEL.RCNN.THRESH_LO) for stage_tresh in cfg.MODEL.RCNN.CASCADE.
        THRESHOLDS]
    box_encoders = [TargetEncoder(bbox_reg_weights=bbox_weights) for
        bbox_weights in cfg.MODEL.RCNN.CASCADE.BBOX_REG_WEIGHTS]
    inference_detector = BoxDetector(use_batched_nms=cfg.MODEL.INFERENCE.
        USE_BATCHED_NMS, rpn_post_nms_topn=cfg.MODEL.INFERENCE.
        POST_NMS_TOPN, detections_per_image=cfg.MODEL.INFERENCE.
        DETECTIONS_PER_IMAGE, test_nms=cfg.MODEL.INFERENCE.DETECTOR_NMS,
        class_agnostic_box=cfg.MODEL.FRCNN.CLASS_AGNOSTIC, bbox_reg_weights
        =cfg.MODEL.RCNN.CASCADE.BBOX_REG_WEIGHTS[-1])
    if cfg.MODEL.INCLUDE_MASK:
        mask_roi_extractor = GenericRoIExtractor(cfg.MODEL.MRCNN.ROI_SIZE,
            cfg.MODEL.MRCNN.GPU_INFERENCE)
        mask_head = build_mask_head(cfg)
    else:
        mask_head = None
        mask_roi_extractor = None
    return roi_head(bbox_head=bbox_heads, bbox_roi_extractor=
        bbox_roi_extractor, bbox_sampler=bbox_samplers, box_encoder=
        box_encoders, inference_detector=inference_detector, stage_weights=
        cfg.MODEL.RCNN.CASCADE.STAGE_WEIGHTS, mask_head=mask_head,
        mask_roi_extractor=mask_roi_extractor)
