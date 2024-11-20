def __init__(self, t=0.0):
    """
        Initialize the Profile class.

        Args:
            t (float): Initial time. Defaults to 0.0.
        """
    self.t = t
    self.cuda = torch.cuda.is_available()
