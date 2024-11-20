def truncate(self, log_p_x_0: torch.Tensor, truncation_rate: float
    ) ->torch.Tensor:
    """
        Truncates `log_p_x_0` such that for each column vector, the total cumulative probability is `truncation_rate`
        The lowest probabilities that would increase the cumulative probability above `truncation_rate` are set to
        zero.
        """
    sorted_log_p_x_0, indices = torch.sort(log_p_x_0, 1, descending=True)
    sorted_p_x_0 = torch.exp(sorted_log_p_x_0)
    keep_mask = sorted_p_x_0.cumsum(dim=1) < truncation_rate
    all_true = torch.full_like(keep_mask[:, 0:1, :], True)
    keep_mask = torch.cat((all_true, keep_mask), dim=1)
    keep_mask = keep_mask[:, :-1, :]
    keep_mask = keep_mask.gather(1, indices.argsort(1))
    rv = log_p_x_0.clone()
    rv[~keep_mask] = -torch.inf
    return rv
