@classmethod
def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
    ret = super().from_config(cfg, input_shape)
    ret['box2box_transform'] = Box2BoxTransformRotated(weights=cfg.MODEL.
        RPN.BBOX_REG_WEIGHTS)
    return ret
