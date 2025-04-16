def measurer_measure(self):
    power = N.nvmlDeviceGetPowerUsage(self.device_handle)
    self.power.append(power)
