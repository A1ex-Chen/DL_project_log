@classmethod
def from_config(cls, cfg):
    ret = super().from_config(cfg)
    ret.update({'combine_overlap_thresh': cfg.MODEL.PANOPTIC_FPN.COMBINE.
        OVERLAP_THRESH, 'combine_stuff_area_thresh': cfg.MODEL.PANOPTIC_FPN
        .COMBINE.STUFF_AREA_LIMIT, 'combine_instances_score_thresh': cfg.
        MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH})
    ret['sem_seg_head'] = build_sem_seg_head(cfg, ret['backbone'].
        output_shape())
    logger = logging.getLogger(__name__)
    if not cfg.MODEL.PANOPTIC_FPN.COMBINE.ENABLED:
        logger.warning(
            'PANOPTIC_FPN.COMBINED.ENABLED is no longer used.  model.inference(do_postprocess=) should be used to toggle postprocessing.'
            )
    if cfg.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT != 1.0:
        w = cfg.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT
        logger.warning(
            'PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT should be replaced by weights on each ROI head.'
            )

        def update_weight(x):
            if isinstance(x, dict):
                return {k: (v * w) for k, v in x.items()}
            else:
                return x * w
        roi_heads = ret['roi_heads']
        roi_heads.box_predictor.loss_weight = update_weight(roi_heads.
            box_predictor.loss_weight)
        roi_heads.mask_head.loss_weight = update_weight(roi_heads.mask_head
            .loss_weight)
    return ret
