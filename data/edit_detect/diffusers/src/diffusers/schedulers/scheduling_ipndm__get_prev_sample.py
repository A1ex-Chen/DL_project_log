def _get_prev_sample(self, sample, timestep_index, prev_timestep_index, ets):
    alpha = self.alphas[timestep_index]
    sigma = self.betas[timestep_index]
    next_alpha = self.alphas[prev_timestep_index]
    next_sigma = self.betas[prev_timestep_index]
    pred = (sample - sigma * ets) / max(alpha, 1e-08)
    prev_sample = next_alpha * pred + ets * next_sigma
    return prev_sample
