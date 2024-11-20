def _prepare_tensor(self, value, device):
    if not torch.is_tensor(value):
        if isinstance(value, float):
            dtype = torch.float32 if device.type == 'mps' else torch.float64
        else:
            dtype = torch.int32 if device.type == 'mps' else torch.int64
        return torch.tensor([value], dtype=dtype, device=device)
    elif len(value.shape) == 0:
        return value[None].to(device)
    else:
        return value
