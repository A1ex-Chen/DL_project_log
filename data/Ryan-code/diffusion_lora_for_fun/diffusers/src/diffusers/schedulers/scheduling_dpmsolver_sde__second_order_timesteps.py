def _second_order_timesteps(self, sigmas, log_sigmas):

    def sigma_fn(_t):
        return np.exp(-_t)

    def t_fn(_sigma):
        return -np.log(_sigma)
    midpoint_ratio = 0.5
    t = t_fn(sigmas)
    delta_time = np.diff(t)
    t_proposed = t[:-1] + delta_time * midpoint_ratio
    sig_proposed = sigma_fn(t_proposed)
    timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in
        sig_proposed])
    return timesteps
