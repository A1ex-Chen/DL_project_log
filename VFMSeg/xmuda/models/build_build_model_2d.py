def build_model_2d(cfg, model_name=' '):
    model = Net2DSeg(num_classes=cfg.MODEL_2D.NUM_CLASSES, backbone_2d=cfg.
        MODEL_2D.TYPE, backbone_2d_kwargs=cfg.MODEL_2D[cfg.MODEL_2D.TYPE],
        dual_head=cfg.MODEL_2D.DUAL_HEAD)
    train_metric = SegIoU(cfg.MODEL_2D.NUM_CLASSES, name='seg_iou_2d' +
        model_name)
    return model, train_metric
