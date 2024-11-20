@HOOKS.register('CocoEvaluator')
def build_coco_evaluator(cfg):
    assert Path(cfg.PATHS.VAL_ANNOTATIONS).exists()
    return CocoEvaluator(build_dataset(cfg, mode='eval'), cfg.PATHS.
        VAL_ANNOTATIONS, per_epoch=cfg.SOLVER.EVAL_EPOCH_EVAL,
        include_mask_head=cfg.MODEL.INCLUDE_MASK)
