def __init__(self, cost_gain=None, use_fl=True, with_mask=False,
    num_sample_points=12544, alpha=0.25, gamma=2.0):
    """Initializes HungarianMatcher with cost coefficients, Focal Loss, mask prediction, sample points, and alpha
        gamma factors.
        """
    super().__init__()
    if cost_gain is None:
        cost_gain = {'class': 1, 'bbox': 5, 'giou': 2, 'mask': 1, 'dice': 1}
    self.cost_gain = cost_gain
    self.use_fl = use_fl
    self.with_mask = with_mask
    self.num_sample_points = num_sample_points
    self.alpha = alpha
    self.gamma = gamma
