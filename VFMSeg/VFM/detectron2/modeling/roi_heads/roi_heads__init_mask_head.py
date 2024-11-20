@classmethod
def _init_mask_head(cls, cfg, input_shape):
    if not cfg.MODEL.MASK_ON:
        return {}
    in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
    pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
    pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
    sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
    pooler_type = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
    in_channels = [input_shape[f].channels for f in in_features][0]
    ret = {'mask_in_features': in_features}
    ret['mask_pooler'] = ROIPooler(output_size=pooler_resolution, scales=
        pooler_scales, sampling_ratio=sampling_ratio, pooler_type=pooler_type
        ) if pooler_type else None
    if pooler_type:
        shape = ShapeSpec(channels=in_channels, width=pooler_resolution,
            height=pooler_resolution)
    else:
        shape = {f: input_shape[f] for f in in_features}
    ret['mask_head'] = build_mask_head(cfg, shape)
    return ret
