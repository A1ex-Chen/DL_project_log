def _gaussian_weights(self, tile_width, tile_height, nbatches):
    """Generates a gaussian mask of weights for tile contributions"""
    import numpy as np
    from numpy import exp, pi, sqrt
    latent_width = tile_width // 8
    latent_height = tile_height // 8
    var = 0.01
    midpoint = (latent_width - 1) / 2
    x_probs = [(exp(-(x - midpoint) * (x - midpoint) / (latent_width *
        latent_width) / (2 * var)) / sqrt(2 * pi * var)) for x in range(
        latent_width)]
    midpoint = latent_height / 2
    y_probs = [(exp(-(y - midpoint) * (y - midpoint) / (latent_height *
        latent_height) / (2 * var)) / sqrt(2 * pi * var)) for y in range(
        latent_height)]
    weights = np.outer(y_probs, x_probs)
    return torch.tile(torch.tensor(weights, device=self.device), (nbatches,
        self.unet.config.in_channels, 1, 1))
