def sigma_to_t(self, sigma):
    log_sigma = sigma.log()
    dists = log_sigma - self.log_sigmas[:, None]
    low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=self.
        log_sigmas.shape[0] - 2)
    high_idx = low_idx + 1
    low = self.log_sigmas[low_idx]
    high = self.log_sigmas[high_idx]
    w = (low - log_sigma) / (low - high)
    w = w.clamp(0, 1)
    t = (1 - w) * low_idx + w * high_idx
    t = t.view(sigma.shape)
    return t
