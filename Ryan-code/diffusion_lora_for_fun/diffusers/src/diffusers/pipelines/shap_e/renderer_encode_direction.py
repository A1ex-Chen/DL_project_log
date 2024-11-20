def encode_direction(position, direction=None):
    if direction is None:
        return torch.zeros_like(posenc_nerf(position, min_deg=0, max_deg=8))
    else:
        return posenc_nerf(direction, min_deg=0, max_deg=8)
