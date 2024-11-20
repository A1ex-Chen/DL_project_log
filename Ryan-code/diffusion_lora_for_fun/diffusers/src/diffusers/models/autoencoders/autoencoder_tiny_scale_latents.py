def scale_latents(self, x: torch.Tensor) ->torch.Tensor:
    """raw latents -> [0, 1]"""
    return x.div(2 * self.latent_magnitude).add(self.latent_shift).clamp(0, 1)
