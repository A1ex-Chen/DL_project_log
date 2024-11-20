def _tiled_encode(self, x: torch.Tensor) ->torch.Tensor:
    """Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output.

        Args:
            x (`torch.Tensor`): Input batch of images.

        Returns:
            `torch.Tensor`: Encoded batch of images.
        """
    sf = self.spatial_scale_factor
    tile_size = self.tile_sample_min_size
    blend_size = int(tile_size * self.tile_overlap_factor)
    traverse_size = tile_size - blend_size
    ti = range(0, x.shape[-2], traverse_size)
    tj = range(0, x.shape[-1], traverse_size)
    blend_masks = torch.stack(torch.meshgrid([torch.arange(tile_size / sf) /
        (blend_size / sf - 1)] * 2, indexing='ij'))
    blend_masks = blend_masks.clamp(0, 1).to(x.device)
    out = torch.zeros(x.shape[0], 4, x.shape[-2] // sf, x.shape[-1] // sf,
        device=x.device)
    for i in ti:
        for j in tj:
            tile_in = x[..., i:i + tile_size, j:j + tile_size]
            tile_out = out[..., i // sf:(i + tile_size) // sf, j // sf:(j +
                tile_size) // sf]
            tile = self.encoder(tile_in)
            h, w = tile.shape[-2], tile.shape[-1]
            blend_mask_i = torch.ones_like(blend_masks[0]
                ) if i == 0 else blend_masks[0]
            blend_mask_j = torch.ones_like(blend_masks[1]
                ) if j == 0 else blend_masks[1]
            blend_mask = blend_mask_i * blend_mask_j
            tile, blend_mask = tile[..., :h, :w], blend_mask[..., :h, :w]
            tile_out.copy_(blend_mask * tile + (1 - blend_mask) * tile_out)
    return out
