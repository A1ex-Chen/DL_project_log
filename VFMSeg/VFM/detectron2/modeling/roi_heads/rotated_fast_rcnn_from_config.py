@classmethod
def from_config(cls, cfg, input_shape):
    args = super().from_config(cfg, input_shape)
    args['box2box_transform'] = Box2BoxTransformRotated(weights=cfg.MODEL.
        ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
    return args
