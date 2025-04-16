def __init__(self, evaluators):
    """
        Args:
            evaluators (list): the evaluators to combine.
        """
    super().__init__()
    self._evaluators = evaluators
