def reset(self):
    for meter in self.meters.values():
        meter.reset()
