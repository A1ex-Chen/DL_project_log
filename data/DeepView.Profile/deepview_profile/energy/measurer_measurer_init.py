def measurer_init(self):
    N.nvmlInit()
    self.device_handle = N.nvmlDeviceGetHandleByIndex(0)
