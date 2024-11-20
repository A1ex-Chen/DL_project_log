@classmethod
def _init_box_head(cls, cfg, input_shape):
    in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
    pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
    pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
    sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
    pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
    cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
    cascade_ious = cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS
    assert len(cascade_bbox_reg_weights) == len(cascade_ious)
    assert cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG, 'CascadeROIHeads only support class-agnostic regression now!'
    assert cascade_ious[0] == cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS[0]
    in_channels = [input_shape[f].channels for f in in_features]
    assert len(set(in_channels)) == 1, in_channels
    in_channels = in_channels[0]
    box_pooler = ROIPooler(output_size=pooler_resolution, scales=
        pooler_scales, sampling_ratio=sampling_ratio, pooler_type=pooler_type)
    pooled_shape = ShapeSpec(channels=in_channels, width=pooler_resolution,
        height=pooler_resolution)
    box_heads, box_predictors, proposal_matchers = [], [], []
    for match_iou, bbox_reg_weights in zip(cascade_ious,
        cascade_bbox_reg_weights):
        box_head = build_box_head(cfg, pooled_shape)
        box_heads.append(box_head)
        box_predictors.append(FastRCNNOutputLayers(cfg, box_head.
            output_shape, box2box_transform=Box2BoxTransform(weights=
            bbox_reg_weights)))
        proposal_matchers.append(Matcher([match_iou], [0, 1],
            allow_low_quality_matches=False))
    return {'box_in_features': in_features, 'box_pooler': box_pooler,
        'box_heads': box_heads, 'box_predictors': box_predictors,
        'proposal_matchers': proposal_matchers}
