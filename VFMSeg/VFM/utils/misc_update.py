def update(self, val, n=1, decay=0):
    self.val = val
    if decay:
        alpha = math.exp(-n / decay)
        self.sum = alpha * self.sum + (1 - alpha) * val * n
        self.count = alpha * self.count + (1 - alpha) * n
    else:
        self.sum += val * n
        self.count += n
    self.avg = self.sum / self.count
