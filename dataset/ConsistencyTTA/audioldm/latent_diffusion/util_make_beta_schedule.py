def make_beta_schedule(schedule, n_timestep, linear_start=0.0001,
    linear_end=0.02, cosine_s=0.008):
    if schedule == 'linear':
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5,
            n_timestep, dtype=torch.float64) ** 2
    elif schedule == 'cosine':
        timesteps = torch.arange(n_timestep + 1, dtype=torch.float64
            ) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)
    elif schedule == 'sqrt_linear':
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=
            torch.float64)
    elif schedule == 'sqrt':
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=
            torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()
