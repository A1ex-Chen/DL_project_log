@staticmethod
def _rms(tensor):
    return tensor.norm(2) / tensor.numel() ** 0.5
