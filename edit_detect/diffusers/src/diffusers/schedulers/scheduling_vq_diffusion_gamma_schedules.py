def gamma_schedules(num_diffusion_timesteps: int, gamma_cum_start=9e-06,
    gamma_cum_end=0.99999):
    """
    Cumulative and non-cumulative gamma schedules.

    See section 4.1.
    """
    ctt = np.arange(0, num_diffusion_timesteps) / (num_diffusion_timesteps - 1
        ) * (gamma_cum_end - gamma_cum_start) + gamma_cum_start
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1 - one_minus_ct
    ctt = np.concatenate((ctt[1:], [0]))
    return ct, ctt
