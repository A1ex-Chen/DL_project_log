def add_time_axis(self, p, t):
    """ Adds time axis to points.

        Args:
            p (tensor): points
            t (tensor): time values
        """
    n_pts = p.shape[1]
    t = t.unsqueeze(1).repeat(1, n_pts, 1)
    p_out = torch.cat([p, t], dim=-1)
    return p_out
