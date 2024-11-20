def _convert_srgb_to_linear(u: torch.Tensor):
    return torch.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)
