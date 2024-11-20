def getName(self):
    return pynvml.nvmlDeviceGetName(self.handle)
