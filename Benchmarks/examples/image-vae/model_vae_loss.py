def vae_loss(self, x_decoded_mean, x):
    z_mean, z_log_var = self.mu, self.log_v
    bce = nn.MSELoss(reduction='sum')
    xent_loss = bce(x_decoded_mean, x.detach())
    kl_loss = -0.5 * torch.mean(1.0 + z_log_var - z_mean ** 2.0 - torch.exp
        (z_log_var))
    return kl_loss + xent_loss
