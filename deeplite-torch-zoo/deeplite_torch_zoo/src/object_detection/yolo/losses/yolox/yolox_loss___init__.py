def __init__(self, model, device='cuda', autobalance=False):
    super(ComputeXLoss, self).__init__()
    det = model.module.model[-1] if is_parallel(model) else model.model[-1]
    self.det = det
    self.device = device
