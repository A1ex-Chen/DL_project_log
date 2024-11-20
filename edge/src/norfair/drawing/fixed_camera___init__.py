def __init__(self, scale: float=2, attenuation: float=0.05):
    self.scale = scale
    self._background = None
    self._attenuation_factor = 1 - attenuation
