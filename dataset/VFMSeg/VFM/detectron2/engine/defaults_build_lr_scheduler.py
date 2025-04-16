@classmethod
def build_lr_scheduler(cls, cfg, optimizer):
    """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
    return build_lr_scheduler(cfg, optimizer)
