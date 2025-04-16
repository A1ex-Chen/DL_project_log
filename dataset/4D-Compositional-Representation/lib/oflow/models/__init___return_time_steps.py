def return_time_steps(self, t):
    """ Returns time steps for the ODE Solver.
        The time steps are ordered, duplicates are removed, and time 0
        is added for the start.

        Args:
            t (tensor): time values
        """
    device = self.device
    t_steps_eval, t_order = torch.unique(torch.cat([torch.zeros(1).to(
        device), t]), sorted=True, return_inverse=True)
    return t_steps_eval, t_order[1:]
