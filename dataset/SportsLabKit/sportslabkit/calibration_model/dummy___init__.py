def __init__(self, homographies, mode='constant'):
    super().__init__()
    self.homographies = homographies
    self.mode = mode
    self.image_count = 0
