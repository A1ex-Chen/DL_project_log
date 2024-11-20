def _len_norm(self, x, epsilon=1e-06):
    """
        length normalization
        """
    variance = torch.mean(x ** 2, -1, keepdim=True)
    norm_x = x * torch.rsqrt(variance + epsilon)
    return norm_x
