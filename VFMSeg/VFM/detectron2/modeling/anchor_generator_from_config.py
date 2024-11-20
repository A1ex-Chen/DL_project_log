@classmethod
def from_config(cls, cfg, input_shape: List[ShapeSpec]):
    return {'sizes': cfg.MODEL.ANCHOR_GENERATOR.SIZES, 'aspect_ratios': cfg
        .MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS, 'strides': [x.stride for x in
        input_shape], 'offset': cfg.MODEL.ANCHOR_GENERATOR.OFFSET, 'angles':
        cfg.MODEL.ANCHOR_GENERATOR.ANGLES}
