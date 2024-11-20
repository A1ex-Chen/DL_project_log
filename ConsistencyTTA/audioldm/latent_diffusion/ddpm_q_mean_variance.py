def q_mean_variance(self, x_start, t):
    """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
    mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape
        ) * x_start
    variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
    log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t,
        x_start.shape)
    return mean, variance, log_variance
