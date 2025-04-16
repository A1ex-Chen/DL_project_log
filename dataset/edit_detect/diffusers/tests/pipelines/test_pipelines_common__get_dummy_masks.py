def _get_dummy_masks(self, input_size: int=64):
    _masks = torch.zeros((1, 1, input_size, input_size), device=torch_device)
    _masks[0, :, :, :int(input_size / 2)] = 1
    return _masks
