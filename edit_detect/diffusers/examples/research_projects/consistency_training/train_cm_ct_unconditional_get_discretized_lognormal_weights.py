def get_discretized_lognormal_weights(noise_levels: torch.Tensor, p_mean:
    float=-1.1, p_std: float=2.0):
    """
    Calculates the unnormalized weights for a 1D array of noise level sigma_i based on the discretized lognormal"
    " distribution used in the iCT paper (given in Equation 10).
    """
    upper_prob = torch.special.erf((torch.log(noise_levels[1:]) - p_mean) /
        (math.sqrt(2) * p_std))
    lower_prob = torch.special.erf((torch.log(noise_levels[:-1]) - p_mean) /
        (math.sqrt(2) * p_std))
    weights = upper_prob - lower_prob
    return weights
