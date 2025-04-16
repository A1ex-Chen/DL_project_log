def hardware_information(nvml):
    hardware_info = {'hostname': platform.node(), 'os': ' '.join(list(
        platform.uname())), 'gpus': nvml.get_device_names()}
    return hardware_info
