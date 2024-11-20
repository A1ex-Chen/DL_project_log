@classmethod
def from_config(cls, cfg, input_shape):
    ret = super().from_config(cfg, input_shape)
    conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
    num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
    ret.update(conv_dims=[conv_dim] * (num_conv + 1), conv_norm=cfg.MODEL.
        ROI_MASK_HEAD.NORM, input_shape=input_shape)
    if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
        ret['num_classes'] = 1
    else:
        ret['num_classes'] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
    return ret
