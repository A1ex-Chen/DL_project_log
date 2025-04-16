def __init__(self, gamma: float=0.5):
    """
        Initialize the ExponentialMovingAverage motion model.

        Args:
            gamma (float): The weight for the exponential moving average calculation. Default is 0.5.
        """
    super().__init__()
    self._value = None
