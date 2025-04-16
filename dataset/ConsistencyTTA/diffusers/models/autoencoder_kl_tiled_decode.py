def tiled_decode(self, z: torch.FloatTensor, return_dict: bool=True) ->Union[
    DecoderOutput, torch.FloatTensor]:
    """Decode a batch of images using a tiled decoder.

        Args:
        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled decoding is:
        different from non-tiled decoding due to each tile using a different decoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        look of the output, but they should be much less noticeable.
            z (`torch.FloatTensor`): Input batch of latent vectors. return_dict (`bool`, *optional*, defaults to
            `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
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
