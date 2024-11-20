@classmethod
def build_optimizer(cls, cfg, model):
    """
        Returns:
            torch.optim.Optimizer:

        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
    return build_optimizer(cfg, model)
