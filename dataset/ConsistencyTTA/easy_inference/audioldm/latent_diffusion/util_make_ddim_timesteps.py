def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps,
    num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = (np.linspace(0, np.sqrt(num_ddpm_timesteps * 0.8),
            num_ddim_timesteps) ** 2).astype(int)
    else:
        raise NotImplementedError(
            f'There is no ddim discretization method called "{ddim_discr_method}"'
            )
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out
