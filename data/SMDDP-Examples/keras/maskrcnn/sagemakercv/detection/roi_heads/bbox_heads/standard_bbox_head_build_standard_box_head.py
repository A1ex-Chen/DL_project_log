@HEADS.register('StandardBBoxHead')
def build_standard_box_head(cfg):
    bbox_head = StandardBBoxHead
    if cfg.MODEL.RCNN.ROI_HEAD == 'CascadeRoIHead':
        assert len(cfg.MODEL.RCNN.CASCADE.BBOX_REG_WEIGHTS) == len(cfg.
            MODEL.RCNN.CASCADE.THRESHOLDS)
        num_stages = len(cfg.MODEL.RCNN.CASCADE.THRESHOLDS)
        return [StandardBBoxHead(num_classes=cfg.INPUT.NUM_CLASSES,
            mlp_head_dim=cfg.MODEL.FRCNN.MLP_DIM, name=f'box_head_{stage}',
            trainable=cfg.MODEL.FRCNN.TRAINABLE, class_agnostic_box=True if
            stage < num_stages - 1 else cfg.MODEL.FRCNN.CLASS_AGNOSTIC,
            loss_cfg=dict(num_classes=cfg.INPUT.NUM_CLASSES, box_loss_type=
            cfg.MODEL.FRCNN.LOSS_TYPE, use_carl_loss=cfg.MODEL.FRCNN.CARL,
            bbox_reg_weights=cfg.MODEL.RCNN.CASCADE.BBOX_REG_WEIGHTS[stage],
            fast_rcnn_box_loss_weight=1, image_size=cfg.INPUT.IMAGE_SIZE,
            class_agnostic_box=True if stage < num_stages - 1 else cfg.
            MODEL.FRCNN.CLASS_AGNOSTIC)) for stage in range(num_stages)]
    return bbox_head(num_classes=cfg.INPUT.NUM_CLASSES, mlp_head_dim=cfg.
        MODEL.FRCNN.MLP_DIM, trainable=cfg.MODEL.FRCNN.TRAINABLE,
        class_agnostic_box=cfg.MODEL.FRCNN.CLASS_AGNOSTIC, loss_cfg=dict(
        num_classes=cfg.INPUT.NUM_CLASSES, box_loss_type=cfg.MODEL.FRCNN.
        LOSS_TYPE, use_carl_loss=cfg.MODEL.FRCNN.CARL, bbox_reg_weights=cfg
        .MODEL.BBOX_REG_WEIGHTS, fast_rcnn_box_loss_weight=cfg.MODEL.FRCNN.
        LOSS_WEIGHT, image_size=cfg.INPUT.IMAGE_SIZE, class_agnostic_box=
        cfg.MODEL.FRCNN.CLASS_AGNOSTIC))
