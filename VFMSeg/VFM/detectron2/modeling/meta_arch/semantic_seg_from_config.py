@classmethod
def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
    return {'input_shape': {k: v for k, v in input_shape.items() if k in
        cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES}, 'ignore_value': cfg.MODEL.
        SEM_SEG_HEAD.IGNORE_VALUE, 'num_classes': cfg.MODEL.SEM_SEG_HEAD.
        NUM_CLASSES, 'conv_dims': cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
        'common_stride': cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE, 'norm': cfg.
        MODEL.SEM_SEG_HEAD.NORM, 'loss_weight': cfg.MODEL.SEM_SEG_HEAD.
        LOSS_WEIGHT}
