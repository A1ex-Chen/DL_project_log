def __init__(self, optimizer, trust_coefficient=0.02, clip=True, eps=1e-08):
    self.param_groups = optimizer.param_groups
    self.optim = optimizer
    self.trust_coefficient = trust_coefficient
    self.eps = eps
    self.clip = clip
