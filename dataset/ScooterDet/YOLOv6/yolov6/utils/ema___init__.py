def __init__(self, model, decay=0.9999, updates=0):
    self.ema = deepcopy(model.module if is_parallel(model) else model).eval()
    self.updates = updates
    self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
    for param in self.ema.parameters():
        param.requires_grad_(False)
