def __init__(self, smoothing=0.0):
    """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
    super(LabelSmoothing, self).__init__()
    self.confidence = 1.0 - smoothing
    self.smoothing = smoothing
