def __init__(self, model, points_batch_size=100000, threshold=0.5, device=
    None, resolution0=16, upsampling_steps=3, padding=0.1):
    self.model = model.to(device)
    self.points_batch_size = points_batch_size
    self.threshold = threshold
    self.device = device
    self.resolution0 = resolution0
    self.upsampling_steps = upsampling_steps
    self.padding = padding
