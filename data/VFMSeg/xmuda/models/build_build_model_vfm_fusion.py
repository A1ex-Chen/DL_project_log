def build_model_vfm_fusion(cfg, device, model_name=' '):
    assert cfg.MODEL_2D.NUM_CLASSES == cfg.MODEL_3D.NUM_CLASSES
    model = LogitsFusion(device, num_classes=cfg.MODEL_2D.NUM_CLASSES)
    train_metric = SegIoU(cfg.MODEL_2D.NUM_CLASSES, name=
        'seg_iou_vfm_fusion' + model_name)
    return model, train_metric
