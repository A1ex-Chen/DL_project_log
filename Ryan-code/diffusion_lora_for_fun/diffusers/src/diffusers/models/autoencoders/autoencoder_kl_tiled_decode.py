def tiled_decode(self, z: torch.Tensor, return_dict: bool=True) ->Union[
    DecoderOutput, torch.Tensor]:
    """
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
    overlap_size = int(self.tile_latent_min_size * (1 - self.
        tile_overlap_factor))
    blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
    row_limit = self.tile_sample_min_size - blend_extent
    rows = []
    for i in range(0, z.shape[2], overlap_size):
        row = []
        for j in range(0, z.shape[3], overlap_size):
            tile = z[:, :, i:i + self.tile_latent_min_size, j:j + self.
                tile_latent_min_size]
            tile = self.post_quant_conv(tile)
            decoded = self.decoder(tile)
            row.append(decoded)
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
    dec = torch.cat(result_rows, dim=2)
    if not return_dict:
        return dec,
    return DecoderOutput(sample=dec)
