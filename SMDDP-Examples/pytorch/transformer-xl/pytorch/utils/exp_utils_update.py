def update(self, val, n=1):
    self.iters += 1
    self.val = val
    if self.iters > self.warmup:
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.keep:
            self.vals.append(val)
