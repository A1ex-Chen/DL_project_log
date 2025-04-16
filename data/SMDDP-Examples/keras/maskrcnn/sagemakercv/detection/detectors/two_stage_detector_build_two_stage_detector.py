@DETECTORS.register('TwoStageDetector')
def build_two_stage_detector(cfg):
    detector = TwoStageDetector
    return detector(backbone=build_backbone(cfg), neck=build_neck(cfg),
        dense_head=build_dense_head(cfg), roi_head=build_roi_head(cfg),
        global_gradient_clip_ratio=cfg.SOLVER.GRADIENT_CLIP_RATIO,
        weight_decay=0.0 if cfg.SOLVER.OPTIMIZER in ['NovoGrad', 'Adam'] else
        cfg.SOLVER.WEIGHT_DECAY)
