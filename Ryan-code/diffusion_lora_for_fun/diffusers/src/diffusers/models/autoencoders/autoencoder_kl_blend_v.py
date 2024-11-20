def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) ->torch.Tensor:
    blend_extent = min(a.shape[2], b.shape[2], blend_extent)
    for y in range(blend_extent):
        b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent
            ) + b[:, :, y, :] * (y / blend_extent)
    return b
