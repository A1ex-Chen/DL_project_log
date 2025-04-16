def __init__(self, device_idx):
    super().__init__()
    self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
