def _sigma_to_alpha_sigma_t(self, sigma):
    alpha_t = 1 / (sigma ** 2 + 1) ** 0.5
    sigma_t = sigma * alpha_t
    return alpha_t, sigma_t
