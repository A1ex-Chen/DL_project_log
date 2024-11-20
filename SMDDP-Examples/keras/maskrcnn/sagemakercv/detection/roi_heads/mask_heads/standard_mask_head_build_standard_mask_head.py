@HEADS.register('StandardMaskHead')
def build_standard_mask_head(cfg):
    mask_head = StandardMaskHead
    return mask_head(num_classes=cfg.INPUT.NUM_CLASSES, mrcnn_resolution=
        cfg.MODEL.MRCNN.RESOLUTION, is_gpu_inference=cfg.MODEL.MRCNN.
        GPU_INFERENCE, trainable=cfg.MODEL.MRCNN.TRAINABLE, loss_cfg=dict(
        mrcnn_weight_loss_mask=cfg.MODEL.MRCNN.LOSS_WEIGHT, label_smoothing
        =cfg.MODEL.MRCNN.LABEL_SMOOTHING))
