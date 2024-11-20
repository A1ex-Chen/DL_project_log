def get_loss_recon(self, data, c_s=None, c_t=None, z=None, z_t=None):
    """ Computes the reconstruction loss.

        Args:
            data (dict): data dictionary
            c_s (tensor): spatial conditioned code
            c_t (tensor): temporal conditioned code
            z (tensor): latent code
            z_t (tensor): latent temporal code
        """
    if not self.loss_recon:
        return 0
    loss_t0 = self.get_loss_recon_t0(data, c_s, z)
    loss_t = self.get_loss_recon_t(data, c_s, c_t, z, z_t)
    return loss_t0 + loss_t
