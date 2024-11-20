def __init__(self, positive_fraction=0.5, is_static=False):
    """Constructs a minibatch sampler.

    Args:
      positive_fraction: desired fraction of positive examples (scalar in [0,1])
        in the batch.
      is_static: If True, uses an implementation with static shape guarantees.

    Raises:
      ValueError: if positive_fraction < 0, or positive_fraction > 1
    """
    if positive_fraction < 0 or positive_fraction > 1:
        raise ValueError(
            'positive_fraction should be in range [0,1]. Received: %s.' %
            positive_fraction)
    self._positive_fraction = positive_fraction
    self._is_static = is_static
