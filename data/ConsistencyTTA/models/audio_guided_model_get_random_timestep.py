def get_random_timestep(batch_size):
    device = self.text_encoder.device
    avail_timesteps = self.noise_scheduler.timesteps.to(device)
    time_inds = torch.randint(0, len(avail_timesteps), (batch_size,))
    t_n = avail_timesteps[time_inds.to(device)]
    return t_n
