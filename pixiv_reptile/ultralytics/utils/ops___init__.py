def __init__(self, t=0.0, device: torch.device=None):
    """
        Initialize the Profile class.

        Args:
            t (float): Initial time. Defaults to 0.0.
            device (torch.device): Devices used for model inference. Defaults to None (cpu).
        """
    self.t = t
    self.device = device
    self.cuda = bool(device and str(device).startswith('cuda'))
