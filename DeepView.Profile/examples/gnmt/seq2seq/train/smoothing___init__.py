def __init__(self, padding_idx, smoothing=0.0):
    """
        Constructor for the LabelSmoothing module.

        :param padding_idx: index of the PAD token
        :param smoothing: label smoothing factor
        """
    super(LabelSmoothing, self).__init__()
    self.padding_idx = padding_idx
    self.confidence = 1.0 - smoothing
    self.smoothing = smoothing
