def get_time_steps(self):
    """ Return time steps values.
        """
    n_steps = self.n_time_steps
    device = self.onet_generator.device
    if self.only_end_time_points:
        t = torch.tensor([0.0, 1.0]).to(device)
    else:
        t = (torch.arange(1, n_steps).float() / (n_steps - 1)).to(device)
    return t
