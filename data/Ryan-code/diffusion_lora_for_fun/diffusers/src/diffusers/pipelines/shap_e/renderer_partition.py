def partition(self, ts):
    """
        Partitions t0 and t1 into n_samples intervals.

        Args:
            ts: [batch_size, *shape, n_samples, 1]

        Return:

            lower: [batch_size, *shape, n_samples, 1] upper: [batch_size, *shape, n_samples, 1] delta: [batch_size,
            *shape, n_samples, 1]

        where
            ts \\in [lower, upper] deltas = upper - lower
        """
    mids = (ts[..., 1:, :] + ts[..., :-1, :]) * 0.5
    lower = torch.cat([self.t0[..., None, :], mids], dim=-2)
    upper = torch.cat([mids, self.t1[..., None, :]], dim=-2)
    delta = upper - lower
    assert lower.shape == upper.shape == delta.shape == ts.shape
    return lower, upper, delta
