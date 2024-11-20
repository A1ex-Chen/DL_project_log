def encode(self, sample, deterministic: bool=True, return_dict: bool=True):
    sample = jnp.transpose(sample, (0, 2, 3, 1))
    hidden_states = self.encoder(sample, deterministic=deterministic)
    moments = self.quant_conv(hidden_states)
    posterior = FlaxDiagonalGaussianDistribution(moments)
    if not return_dict:
        return posterior,
    return FlaxAutoencoderKLOutput(latent_dist=posterior)
