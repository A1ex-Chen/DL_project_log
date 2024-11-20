def _get_device_type(torch_tensor):
    assert torch_tensor.device.type in ['cpu', 'cuda']
    assert torch_tensor.device.index == 0
    return torch_tensor.device.type
