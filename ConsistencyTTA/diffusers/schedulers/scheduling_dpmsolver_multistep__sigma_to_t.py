def _sigma_to_t(self, sigma, log_sigmas):
    log_sigma = np.log(sigma)
    dists = log_sigma - log_sigmas[:, np.newaxis]
    low_idx = np.cumsum(dists >= 0, axis=0).argmax(axis=0).clip(max=
        log_sigmas.shape[0] - 2)
    high_idx = low_idx + 1
    low = log_sigmas[low_idx]
    high = log_sigmas[high_idx]
    w = (low - log_sigma) / (low - high)
    w = np.clip(w, 0, 1)
    t = (1 - w) * low_idx + w * high_idx
    t = t.reshape(sigma.shape)
    return t
