def unscale_latents(self, x: torch.Tensor) ->torch.Tensor:
    """[0, 1] -> raw latents"""
    return x.sub(self.latent_shift).mul(2 * self.latent_magnitude)
