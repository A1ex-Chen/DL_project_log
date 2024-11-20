def __init__(self, model):
    """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
    self.one2many = v8DetectionLoss(model, tal_topk=10)
    self.one2one = v8DetectionLoss(model, tal_topk=1)
