def comp_unrolled_model(self, data, target, eta, optimizer):
    """Loss on train set and then update w_pi, not-in-place

        Parameters
        ----------
        data : torch.tensor

        target : torch.tensor
        eta : float
        optimizer : torch.optim.optimizer
             optimizer of theta, not optimizer of alpha

        Returns
        -------
        model_unrolled
        """
    loss = self.model.loss(data, target)
    theta = F.flatten(self.model.parameters()).detach()
    try:
        moment = F.flatten(optimizer.state[v]['momentum_buffer'] for v in
            self.model.parameters())
        moment.mul_(self.momentum)
    except Exception:
        moment = torch.zeros_like(theta)
    dtheta = F.flatten(autograd.grad(loss, self.model.parameters())).data
    theta = theta.sub(eta, moment + dtheta + self.wd * theta)
    unrolled_model = self.construct_model_from_theta(theta)
    return unrolled_model.to(self.device)
