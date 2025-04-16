def __init__(self, obs_steps: int=25):
    """
        Initialize the ExponentialMovingAverage motion model.

        """
    super().__init__()
    self.model = Linear(obs_steps=obs_steps)
