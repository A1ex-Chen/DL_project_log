@classmethod
def from_config(cls, cfg, input_shape):
    ret = super().from_config(cfg)
    ret['train_on_pred_boxes'] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
    if inspect.ismethod(cls._init_box_head):
        ret.update(cls._init_box_head(cfg, input_shape))
    if inspect.ismethod(cls._init_mask_head):
        ret.update(cls._init_mask_head(cfg, input_shape))
    if inspect.ismethod(cls._init_keypoint_head):
        ret.update(cls._init_keypoint_head(cfg, input_shape))
    return ret
