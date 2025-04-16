def get_coefficients_exponential_positive(self, order, interval_start,
    interval_end, tau):
    """
        Calculate the integral of exp(x(1+tau^2)) * x^order dx from interval_start to interval_end
        """
    assert order in [0, 1, 2, 3], 'order is only supported for 0, 1, 2 and 3'
    interval_end_cov = (1 + tau ** 2) * interval_end
    interval_start_cov = (1 + tau ** 2) * interval_start
    if order == 0:
        return torch.exp(interval_end_cov) * (1 - torch.exp(-(
            interval_end_cov - interval_start_cov))) / (1 + tau ** 2)
    elif order == 1:
        return torch.exp(interval_end_cov) * (interval_end_cov - 1 - (
            interval_start_cov - 1) * torch.exp(-(interval_end_cov -
            interval_start_cov))) / (1 + tau ** 2) ** 2
    elif order == 2:
        return torch.exp(interval_end_cov) * (interval_end_cov ** 2 - 2 *
            interval_end_cov + 2 - (interval_start_cov ** 2 - 2 *
            interval_start_cov + 2) * torch.exp(-(interval_end_cov -
            interval_start_cov))) / (1 + tau ** 2) ** 3
    elif order == 3:
        return torch.exp(interval_end_cov) * (interval_end_cov ** 3 - 3 * 
            interval_end_cov ** 2 + 6 * interval_end_cov - 6 - (
            interval_start_cov ** 3 - 3 * interval_start_cov ** 2 + 6 *
            interval_start_cov - 6) * torch.exp(-(interval_end_cov -
            interval_start_cov))) / (1 + tau ** 2) ** 4
