@TRAINERS.register('DetectionTrainer')
def build_detection_trainer(cfg, model, optimizer, dist=None):
    return DetectionTrainer(model=model, optimizer=optimizer, dist=dist,
        fp16=cfg.SOLVER.FP16, global_gradient_clip_ratio=cfg.SOLVER.
        GRADIENT_CLIP_RATIO, weight_decay=0.0 if cfg.SOLVER.OPTIMIZER in [
        'NovoGrad', 'Adam'] else cfg.SOLVER.WEIGHT_DECAY)
