def to(self, device):
    self.pixel_values.to(device)
    return self
