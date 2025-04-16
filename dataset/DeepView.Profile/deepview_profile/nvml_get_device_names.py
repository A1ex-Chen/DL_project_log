def get_device_names(self):
    device_names = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        device_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
        device_names.append(device_name)
    return device_names
