def __init__(self, c1, c2, k, s):
    """
        Spatial Channel Downsample (SCDown) module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for the convolutional layer.
            s (int): Stride for the convolutional layer.
        """
    super().__init__()
    self.cv1 = Conv(c1, c2, 1, 1)
    self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)
