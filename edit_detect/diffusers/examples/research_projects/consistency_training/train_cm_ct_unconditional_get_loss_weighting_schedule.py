def get_loss_weighting_schedule(noise_levels: torch.Tensor):
    """
    Calculates the loss weighting schedule lambda given a set of noise levels.
    """
    return 1.0 / (noise_levels[1:] - noise_levels[:-1])
