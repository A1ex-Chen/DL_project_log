def compute_snr(self, timesteps, t_indices):
    if self.use_edm:
        return self.noise_scheduler.sigmas[t_indices] ** -2
    else:
        return super().compute_snr(timesteps)
