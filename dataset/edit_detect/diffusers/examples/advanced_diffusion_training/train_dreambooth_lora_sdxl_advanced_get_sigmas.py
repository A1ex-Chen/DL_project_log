def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in
        timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma
