@classmethod
def from_config(cls, cfg, input_shape):
    return {'input_shape': input_shape, 'box2box_transform':
        Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
        'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
        'cls_agnostic_bbox_reg': cfg.MODEL.ROI_BOX_HEAD.
        CLS_AGNOSTIC_BBOX_REG, 'smooth_l1_beta': cfg.MODEL.ROI_BOX_HEAD.
        SMOOTH_L1_BETA, 'test_score_thresh': cfg.MODEL.ROI_HEADS.
        SCORE_THRESH_TEST, 'test_nms_thresh': cfg.MODEL.ROI_HEADS.
        NMS_THRESH_TEST, 'test_topk_per_image': cfg.TEST.
        DETECTIONS_PER_IMAGE, 'box_reg_loss_type': cfg.MODEL.ROI_BOX_HEAD.
        BBOX_REG_LOSS_TYPE, 'loss_weight': {'loss_box_reg': cfg.MODEL.
        ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT}}
