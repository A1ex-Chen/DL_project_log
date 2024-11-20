def build_model_3d(cfg, model_name=' '):
    model = Net3DSeg(num_classes=cfg.MODEL_3D.NUM_CLASSES, backbone_3d=cfg.
        MODEL_3D.TYPE, backbone_3d_kwargs=cfg.MODEL_3D[cfg.MODEL_3D.TYPE],
        dual_head=cfg.MODEL_3D.DUAL_HEAD)
    train_metric = SegIoU(cfg.MODEL_3D.NUM_CLASSES, name='seg_iou_3d' +
        model_name)
    return model, train_metric
