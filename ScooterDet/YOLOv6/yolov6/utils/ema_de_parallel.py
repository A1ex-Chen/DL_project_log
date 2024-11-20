def de_parallel(model):
    """De-parallelize a model. Return single-GPU model if model's type is DP or DDP."""
    return model.module if is_parallel(model) else model
