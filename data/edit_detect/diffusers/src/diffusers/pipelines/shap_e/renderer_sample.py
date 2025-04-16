@torch.no_grad()
def sample(self, t0: torch.Tensor, t1: torch.Tensor, n_samples: int
    ) ->torch.Tensor:
    """
        Args:
            t0: start time has shape [batch_size, *shape, 1]
            t1: finish time has shape [batch_size, *shape, 1]
            n_samples: number of ts to sample
        Return:
            sampled ts of shape [batch_size, *shape, n_samples, 1]
        """
    lower, upper, _ = self.volume_range.partition(self.ts)
    batch_size, *shape, n_coarse_samples, _ = self.ts.shape
    weights = self.weights
    if self.blur_pool:
        padded = torch.cat([weights[..., :1, :], weights, weights[..., -1:,
            :]], dim=-2)
        maxes = torch.maximum(padded[..., :-1, :], padded[..., 1:, :])
        weights = 0.5 * (maxes[..., :-1, :] + maxes[..., 1:, :])
    weights = weights + self.alpha
    pmf = weights / weights.sum(dim=-2, keepdim=True)
    inds = sample_pmf(pmf, n_samples)
    assert inds.shape == (batch_size, *shape, n_samples, 1)
    assert (inds >= 0).all() and (inds < n_coarse_samples).all()
    t_rand = torch.rand(inds.shape, device=inds.device)
    lower_ = torch.gather(lower, -2, inds)
    upper_ = torch.gather(upper, -2, inds)
    ts = lower_ + (upper_ - lower_) * t_rand
    ts = torch.sort(ts, dim=-2).values
    return ts
