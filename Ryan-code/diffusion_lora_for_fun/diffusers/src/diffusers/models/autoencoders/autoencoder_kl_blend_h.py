def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) ->torch.Tensor:
    blend_extent = min(a.shape[3], b.shape[3], blend_extent)
    for x in range(blend_extent):
        b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent
            ) + b[:, :, :, x] * (x / blend_extent)
    return b
