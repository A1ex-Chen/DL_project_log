@classmethod
def from_config(cls, cfg, input_shape: List[ShapeSpec]):
    num_anchors = build_anchor_generator(cfg, input_shape).num_cell_anchors
    assert len(set(num_anchors)
        ) == 1, 'Using different number of anchors between levels is not currently supported!'
    num_anchors = num_anchors[0]
    return {'input_shape': input_shape, 'num_classes': cfg.MODEL.RETINANET.
        NUM_CLASSES, 'conv_dims': [input_shape[0].channels] * cfg.MODEL.
        RETINANET.NUM_CONVS, 'prior_prob': cfg.MODEL.RETINANET.PRIOR_PROB,
        'norm': cfg.MODEL.RETINANET.NORM, 'num_anchors': num_anchors}
