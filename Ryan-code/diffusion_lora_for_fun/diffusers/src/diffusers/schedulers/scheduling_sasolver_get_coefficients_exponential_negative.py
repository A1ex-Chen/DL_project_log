def get_coefficients_exponential_negative(self, order, interval_start,
    interval_end):
    """
        Calculate the integral of exp(-x) * x^order dx from interval_start to interval_end
        """
    assert order in [0, 1, 2, 3], 'order is only supported for 0, 1, 2 and 3'
    if order == 0:
        return torch.exp(-interval_end) * (torch.exp(interval_end -
            interval_start) - 1)
    elif order == 1:
        return torch.exp(-interval_end) * ((interval_start + 1) * torch.exp
            (interval_end - interval_start) - (interval_end + 1))
    elif order == 2:
        return torch.exp(-interval_end) * ((interval_start ** 2 + 2 *
            interval_start + 2) * torch.exp(interval_end - interval_start) -
            (interval_end ** 2 + 2 * interval_end + 2))
    elif order == 3:
        return torch.exp(-interval_end) * ((interval_start ** 3 + 3 * 
            interval_start ** 2 + 6 * interval_start + 6) * torch.exp(
            interval_end - interval_start) - (interval_end ** 3 + 3 * 
            interval_end ** 2 + 6 * interval_end + 6))
