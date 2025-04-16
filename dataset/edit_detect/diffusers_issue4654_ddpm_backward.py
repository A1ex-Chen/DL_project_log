def ddpm_backward(x0, model, scheduler, timesteps=500):
    scheduler.set_timesteps(timesteps)
    sample_size = model.config.sample_size
    noise = torch.tensor(x0).to('cuda')
    input = noise
    for t in tqdm(scheduler.timesteps):
        with torch.no_grad():
            noisy_residual = model(input, t).sample
        previous_noisy_sample = scheduler.step(noisy_residual, t, input
            ).prev_sample
        input = previous_noisy_sample
    return input
