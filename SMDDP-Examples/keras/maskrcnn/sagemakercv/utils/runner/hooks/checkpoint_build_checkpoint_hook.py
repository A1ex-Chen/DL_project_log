@HOOKS.register('CheckpointHook')
def build_checkpoint_hook(cfg):
    return CheckpointHook(interval=cfg.SOLVER.CHECKPOINT_INTERVAL,
        backbone_checkpoint=cfg.PATHS.WEIGHTS)
