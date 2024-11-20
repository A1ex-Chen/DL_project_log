def ddpm_forward(noise, model, scheduler, timesteps=500):
    scheduler.set_timesteps(timesteps)
    sample_size = model.config.sample_size
    input = torch.tensor(noise).to('cuda')
    for t in tqdm(scheduler.timesteps):
        with torch.no_grad():
            noisy_residual = model(input, t).sample
        previous_noisy_sample = scheduler.step(noisy_residual, t, input
            ).prev_sample
        input = previous_noisy_sample
    return input
