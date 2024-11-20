@HEADS.register('StandardRoIHead')
def build_standard_roi_head(cfg):
    roi_head = StandardRoIHead
    bbox_head = build_box_head(cfg)
    bbox_roi_extractor = GenericRoIExtractor(cfg.MODEL.FRCNN.ROI_SIZE, cfg.
        MODEL.FRCNN.GPU_INFERENCE)
    bbox_sampler = RandomSampler(batch_size_per_im=cfg.MODEL.RCNN.
        BATCH_SIZE_PER_IMAGE, fg_fraction=cfg.MODEL.RCNN.FG_FRACTION,
        fg_thresh=cfg.MODEL.RCNN.THRESH, bg_thresh_hi=cfg.MODEL.RCNN.
        THRESH_HI, bg_thresh_lo=cfg.MODEL.RCNN.THRESH_LO)
    box_encoder = TargetEncoder(bbox_reg_weights=cfg.MODEL.BBOX_REG_WEIGHTS)
    inference_detector = BoxDetector(use_batched_nms=cfg.MODEL.INFERENCE.
        USE_BATCHED_NMS, rpn_post_nms_topn=cfg.MODEL.INFERENCE.
        POST_NMS_TOPN, detections_per_image=cfg.MODEL.INFERENCE.
        DETECTIONS_PER_IMAGE, test_nms=cfg.MODEL.INFERENCE.DETECTOR_NMS,
        class_agnostic_box=cfg.MODEL.INFERENCE.CLASS_AGNOSTIC,
        bbox_reg_weights=cfg.MODEL.BBOX_REG_WEIGHTS)
    if cfg.MODEL.INCLUDE_MASK:
        mask_roi_extractor = GenericRoIExtractor(cfg.MODEL.MRCNN.ROI_SIZE,
            cfg.MODEL.MRCNN.GPU_INFERENCE)
        mask_head = build_mask_head(cfg)
    else:
        mask_head = None
        mask_roi_extractor = None
    return roi_head(bbox_head=bbox_head, bbox_roi_extractor=
        bbox_roi_extractor, bbox_sampler=bbox_sampler, box_encoder=
        box_encoder, inference_detector=inference_detector, mask_head=
        mask_head, mask_roi_extractor=mask_roi_extractor)
