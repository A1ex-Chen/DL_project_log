def __init__(self, size, device=False):
    self.size = size
    self._done = False
    self.scores = torch.zeros((size,), dtype=torch.float, device=device)
    self.all_scores = []
    self.prev_ks = []
    self.next_ys = [torch.full((size,), Constants.PAD, dtype=torch.long,
        device=device)]
    self.next_ys[0][0] = Constants.BOS
