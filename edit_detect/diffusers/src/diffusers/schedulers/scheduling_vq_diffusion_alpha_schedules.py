def alpha_schedules(num_diffusion_timesteps: int, alpha_cum_start=0.99999,
    alpha_cum_end=9e-06):
    """
    Cumulative and non-cumulative alpha schedules.

    See section 4.1.
    """
    att = np.arange(0, num_diffusion_timesteps) / (num_diffusion_timesteps - 1
        ) * (alpha_cum_end - alpha_cum_start) + alpha_cum_start
    att = np.concatenate(([1], att))
    at = att[1:] / att[:-1]
    att = np.concatenate((att[1:], [1]))
    return at, att
