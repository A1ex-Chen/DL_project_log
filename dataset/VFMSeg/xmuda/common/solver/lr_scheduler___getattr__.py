def __getattr__(self, item):
    if hasattr(self.scheduler, item):
        return getattr(self.scheduler, item)
    else:
        return getattr(self, item)
