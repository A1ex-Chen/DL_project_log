def _decode(self, z: torch.FloatTensor, return_dict: bool=True) ->Union[
    DecoderOutput, torch.FloatTensor]:
    if self.use_tiling and (z.shape[-1] > self.tile_latent_min_size or z.
        shape[-2] > self.tile_latent_min_size):
        return self.tiled_decode(z, return_dict=return_dict)
    z = self.post_quant_conv(z)
    dec = self.decoder(z)
    if not return_dict:
        return dec,
    return DecoderOutput(sample=dec)
