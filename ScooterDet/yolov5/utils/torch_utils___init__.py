def __init__(self, model, decay=0.9999, tau=2000, updates=0):
    self.ema = deepcopy(de_parallel(model)).eval()
    self.updates = updates
    self.decay = lambda x: decay * (1 - math.exp(-x / tau))
    for p in self.ema.parameters():
        p.requires_grad_(False)
