def sample_pmf(pmf: torch.Tensor, n_samples: int) ->torch.Tensor:
    """
    Sample from the given discrete probability distribution with replacement.

    The i-th bin is assumed to have mass pmf[i].

    Args:
        pmf: [batch_size, *shape, n_samples, 1] where (pmf.sum(dim=-2) == 1).all()
        n_samples: number of samples

    Return:
        indices sampled with replacement
    """
    *shape, support_size, last_dim = pmf.shape
    assert last_dim == 1
    cdf = torch.cumsum(pmf.view(-1, support_size), dim=1)
    inds = torch.searchsorted(cdf, torch.rand(cdf.shape[0], n_samples,
        device=cdf.device))
    return inds.view(*shape, n_samples, 1).clamp(0, support_size - 1)
