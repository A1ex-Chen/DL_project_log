def __init__(self, smoothing=0.0):
    super(NLLMultiLabelSmooth, self).__init__()
    self.confidence = 1.0 - smoothing
    self.smoothing = smoothing
