def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True
    ):
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]
        ].tolist())
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas /
        alphas_prev))
    if verbose:
        print(
            f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}'
            )
        print(
            f'For the chosen value of eta, which is {eta}, this results in the following sigma_t schedule for ddim sampler {sigmas}'
            )
    return sigmas, alphas, alphas_prev
