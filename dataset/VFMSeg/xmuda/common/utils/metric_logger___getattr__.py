def __getattr__(self, attr):
    if attr in self.meters:
        return self.meters[attr]
    return getattr(self, attr)
