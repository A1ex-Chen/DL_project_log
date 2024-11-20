@classmethod
def _init_box_head(cls, cfg, input_shape):
    in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
    pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
    pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
    sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
    pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
    assert pooler_type in ['ROIAlignRotated'], pooler_type
    in_channels = [input_shape[f].channels for f in in_features][0]
    box_pooler = ROIPooler(output_size=pooler_resolution, scales=
        pooler_scales, sampling_ratio=sampling_ratio, pooler_type=pooler_type)
    box_head = build_box_head(cfg, ShapeSpec(channels=in_channels, height=
        pooler_resolution, width=pooler_resolution))
    box_predictor = RotatedFastRCNNOutputLayers(cfg, box_head.output_shape)
    return {'box_in_features': in_features, 'box_pooler': box_pooler,
        'box_head': box_head, 'box_predictor': box_predictor}
