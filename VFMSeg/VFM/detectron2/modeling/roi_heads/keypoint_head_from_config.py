@classmethod
def from_config(cls, cfg, input_shape):
    ret = super().from_config(cfg, input_shape)
    ret['input_shape'] = input_shape
    ret['conv_dims'] = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS
    return ret
