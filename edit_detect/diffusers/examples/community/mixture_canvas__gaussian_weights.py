def _gaussian_weights(self, region: DiffusionRegion) ->torch.tensor:
    """Generates a gaussian mask of weights for tile contributions"""
    latent_width = region.latent_col_end - region.latent_col_init
    latent_height = region.latent_row_end - region.latent_row_init
    var = 0.01
    midpoint = (latent_width - 1) / 2
    x_probs = [(exp(-(x - midpoint) * (x - midpoint) / (latent_width *
        latent_width) / (2 * var)) / sqrt(2 * pi * var)) for x in range(
        latent_width)]
    midpoint = (latent_height - 1) / 2
    y_probs = [(exp(-(y - midpoint) * (y - midpoint) / (latent_height *
        latent_height) / (2 * var)) / sqrt(2 * pi * var)) for y in range(
        latent_height)]
    weights = np.outer(y_probs, x_probs) * region.mask_weight
    return torch.tile(torch.tensor(weights), (self.nbatch, self.
        latent_space_dim, 1, 1))
