def __init__(self, alpha, bootstrap_type='soft'):
    """Constructor.

    Args:
      alpha: a float32 scalar tensor between 0 and 1 representing interpolation
        weight
      bootstrap_type: set to either 'hard' or 'soft' (default)

    Raises:
      ValueError: if bootstrap_type is not either 'hard' or 'soft'
    """
    if bootstrap_type != 'hard' and bootstrap_type != 'soft':
        raise ValueError(
            "Unrecognized bootstrap_type: must be one of 'hard' or 'soft.'")
    self._alpha = alpha
    self._bootstrap_type = bootstrap_type
