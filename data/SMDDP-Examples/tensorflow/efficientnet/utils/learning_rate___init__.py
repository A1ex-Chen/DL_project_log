def __init__(self, batch_size: int, epoch_size: int, warmup_epochs: int,
    boundaries: List[int], multipliers: List[float]):
    """Piecewise constant decay with warmup.

    Args:
      batch_size: The training batch size used in the experiment.
      epoch_size: The size of an epoch, or the number of examples in an epoch.
      warmup_epochs: The number of warmup epochs to apply.
      boundaries: The list of floats with strictly increasing entries.
      multipliers: The list of multipliers/learning rates to use for the
        piecewise portion. The length must be 1 less than that of boundaries.

    """
    super(PiecewiseConstantDecayWithWarmup, self).__init__()
    if len(boundaries) != len(multipliers) - 1:
        raise ValueError(
            'The length of boundaries must be 1 less than the length of multipliers'
            )
    base_lr_batch_size = 256
    steps_per_epoch = epoch_size // batch_size
    self._rescaled_lr = BASE_LEARNING_RATE * batch_size / base_lr_batch_size
    self._step_boundaries = [(float(steps_per_epoch) * x) for x in boundaries]
    self._lr_values = [(self._rescaled_lr * m) for m in multipliers]
    self._warmup_steps = warmup_epochs * steps_per_epoch
