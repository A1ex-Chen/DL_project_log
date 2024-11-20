def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-09):
    """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
    super().__init__()
    self.topk = topk
    self.num_classes = num_classes
    self.bg_idx = num_classes
    self.alpha = alpha
    self.beta = beta
    self.eps = eps
