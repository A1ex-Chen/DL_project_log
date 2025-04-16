def get_random_timestep(batch_size, validation_mode):
    device = self.text_encoder.device
    avail_timesteps = self.noise_scheduler.timesteps.to(device)
    order = 2 if self.use_edm else 1
    if validation_mode != 0:
        time_ind = len(avail_timesteps) - 1 - int(validation_mode * order)
        assert time_ind >= 0
        time_inds = time_ind * torch.ones((batch_size,), dtype=torch.int32,
            device=device)
    else:
        time_inds = torch.randint(0, (len(avail_timesteps) - 1) // order, (
            batch_size,), device=device) * order
    t_nplus1 = avail_timesteps[time_inds]
    t_n = avail_timesteps[time_inds + order]
    return t_nplus1, t_n, time_inds, time_inds + order
