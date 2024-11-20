def concat_time_axis(self, t, p, t_batch=False, invert=False):
    """ Concatenates the time axis to the points tenor.

        Args:
            t (tensor); time values
            p (tensor): points
            t_batch (tensor): time help tensor for batch processing
            invert (bool): whether to go backwards
        """
    batch_size, n_points, _ = p.shape
    t = t.repeat(batch_size)
    if t_batch is not None:
        assert len(t_batch) == batch_size
        if invert:
            t = t_batch - t
        else:
            t = t_batch + t
    t = t.view(batch_size, 1, 1).expand(batch_size, n_points, 1)
    p_out = torch.cat([p, t], dim=-1)
    assert p_out.shape[-1] == self.in_dim
    return p_out
