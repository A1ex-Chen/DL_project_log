def _constant_weights(self, region: DiffusionRegion) ->torch.tensor:
    """Computes a tensor of constant for a given diffusion region"""
    latent_width = region.latent_col_end - region.latent_col_init
    latent_height = region.latent_row_end - region.latent_row_init
    return torch.ones(self.nbatch, self.latent_space_dim, latent_height,
        latent_width) * region.mask_weight
