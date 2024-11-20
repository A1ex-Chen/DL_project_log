def __init__(self, l2_norm_clip, noise_multiplier, num_microbatches=None, *
    args, **kwargs):
    """Initialize the DPOptimizerClass.

      Args:
        l2_norm_clip: Clipping norm (max L2 norm of per microbatch gradients).
        noise_multiplier: Ratio of the standard deviation to the clipping norm.
        num_microbatches: Number of microbatches into which each minibatch is
          split. If `None`, will default to the size of the minibatch, and
          per-example gradients will be computed.
        *args: These will be passed on to the base class `__init__` method.
        **kwargs: These will be passed on to the base class `__init__` method.
      """
    super().__init__(*args, **kwargs)
    self._l2_norm_clip = l2_norm_clip
    self._noise_multiplier = noise_multiplier
    self._num_microbatches = num_microbatches
    self._was_compute_gradients_called = False
