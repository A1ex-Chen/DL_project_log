def decode(self, latents, deterministic: bool=True, return_dict: bool=True):
    if latents.shape[-1] != self.config.latent_channels:
        latents = jnp.transpose(latents, (0, 2, 3, 1))
    hidden_states = self.post_quant_conv(latents)
    hidden_states = self.decoder(hidden_states, deterministic=deterministic)
    hidden_states = jnp.transpose(hidden_states, (0, 3, 1, 2))
    if not return_dict:
        return hidden_states,
    return FlaxDecoderOutput(sample=hidden_states)
