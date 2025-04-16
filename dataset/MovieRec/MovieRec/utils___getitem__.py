def __getitem__(self, key):
    if key not in self.meters:
        meter = AverageMeter()
        meter.update(0)
        return meter
    return self.meters[key]
