def _quartic_weights(self, region: DiffusionRegion) ->torch.tensor:
    """Generates a quartic mask of weights for tile contributions

        The quartic kernel has bounded support over the diffusion region, and a smooth decay to the region limits.
        """
    quartic_constant = 15.0 / 16.0
    support = (np.array(range(region.latent_col_init, region.latent_col_end
        )) - region.latent_col_init) / (region.latent_col_end - region.
        latent_col_init - 1) * 1.99 - 1.99 / 2.0
    x_probs = quartic_constant * np.square(1 - np.square(support))
    support = (np.array(range(region.latent_row_init, region.latent_row_end
        )) - region.latent_row_init) / (region.latent_row_end - region.
        latent_row_init - 1) * 1.99 - 1.99 / 2.0
    y_probs = quartic_constant * np.square(1 - np.square(support))
    weights = np.outer(y_probs, x_probs) * region.mask_weight
    return torch.tile(torch.tensor(weights), (self.nbatch, self.
        latent_space_dim, 1, 1))
