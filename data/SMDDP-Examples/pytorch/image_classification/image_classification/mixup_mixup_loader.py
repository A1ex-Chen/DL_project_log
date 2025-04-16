def mixup_loader(self, loader):
    for input, target in loader:
        i, t = mixup(self.alpha, input, target)
        yield i, t
