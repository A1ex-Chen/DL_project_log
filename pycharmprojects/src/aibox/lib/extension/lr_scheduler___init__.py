def __init__(self, optimizer: Optimizer, milestones: List[int], gamma:
    float=0.1, factor: float=0.3333, num_iters: int=500, last_epoch: int=0):
    self.milestones = milestones
    self.gamma = gamma
    self.factor = factor
    self.num_iters = num_iters
    self.last_warm_iter = 0
    super().__init__(optimizer, last_epoch - 1)
