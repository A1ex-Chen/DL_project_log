@apply_forward_hook
def encode(self, x: torch.Tensor, return_dict: bool=True) ->Union[
    AutoencoderKLOutput, Tuple[torch.Tensor]]:
    h = self.encoder(x)
    moments = self.quant_conv(h)
    posterior = DiagonalGaussianDistribution(moments)
    if not return_dict:
        return posterior,
    return AutoencoderKLOutput(latent_dist=posterior)
