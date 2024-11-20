def compute_kl(self, q_z):
    """ Compute the KL-divergence for predicted and prior distribution.

        Args:
            q_z (dist): predicted distribution
        """
    if q_z.mean.shape[-1] != 0:
        loss_kl = self.vae_beta * dist.kl_divergence(q_z, self.model.p0_z
            ).mean()
        if torch.isnan(loss_kl):
            loss_kl = torch.tensor([0.0]).to(self.device)
    else:
        loss_kl = torch.tensor([0.0]).to(self.device)
    return loss_kl
