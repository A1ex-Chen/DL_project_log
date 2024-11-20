def __init__(self, normalize_by_size=True):
    """
        Args:
            normalize_by_size: normalize deltas by the size of src (anchor) boxes.
        """
    self.normalize_by_size = normalize_by_size
