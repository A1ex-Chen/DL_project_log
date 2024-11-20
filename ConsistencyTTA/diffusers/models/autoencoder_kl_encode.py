@apply_forward_hook
def encode(self, x: torch.FloatTensor, return_dict: bool=True
    ) ->AutoencoderKLOutput:
    if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.
        shape[-2] > self.tile_sample_min_size):
        return self.tiled_encode(x, return_dict=return_dict)
    h = self.encoder(x)
    moments = self.quant_conv(h)
    posterior = DiagonalGaussianDistribution(moments)
    if not return_dict:
        return posterior,
    return AutoencoderKLOutput(latent_dist=posterior)
