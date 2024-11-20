def tiled_encode(self, x: torch.Tensor, return_dict: bool=True) ->Union[
    ConsistencyDecoderVAEOutput, Tuple]:
    """Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.consistency_decoder_vae.ConsistencyDecoderVAEOutput`] instead of a
                plain tuple.

        Returns:
            [`~models.consistency_decoder_vae.ConsistencyDecoderVAEOutput`] or `tuple`:
                If return_dict is True, a [`~models.consistency_decoder_vae.ConsistencyDecoderVAEOutput`] is returned,
                otherwise a plain `tuple` is returned.
        """
    overlap_size = int(self.tile_sample_min_size * (1 - self.
        tile_overlap_factor))
    blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
    row_limit = self.tile_latent_min_size - blend_extent
    rows = []
    for i in range(0, x.shape[2], overlap_size):
        row = []
        for j in range(0, x.shape[3], overlap_size):
            tile = x[:, :, i:i + self.tile_sample_min_size, j:j + self.
                tile_sample_min_size]
            tile = self.encoder(tile)
            tile = self.quant_conv(tile)
            row.append(tile)
        rows.append(row)
    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
            if j > 0:
                tile = self.blend_h(row[j - 1], tile, blend_extent)
            result_row.append(tile[:, :, :row_limit, :row_limit])
        result_rows.append(torch.cat(result_row, dim=3))
    moments = torch.cat(result_rows, dim=2)
    posterior = DiagonalGaussianDistribution(moments)
    if not return_dict:
        return posterior,
    return ConsistencyDecoderVAEOutput(latent_dist=posterior)
