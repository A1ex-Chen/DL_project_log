def get_loss_recon_t0(self, data, c_s=None, z=None):
    """ Computes the reconstruction loss for time step t=0.

        Args:
            data (dict): data dictionary
            c_s (tensor): spatial conditioned code c_s
            z (tensor): latent code z
        """
    p_t0 = data.get('points')
    occ_t0 = data.get('points.occ')
    device = self.device
    batch_size = p_t0.shape[0]
    logits_t0 = self.model.decode(p_t0.to(device), c=c_s, z=z).logits
    loss_occ_t0 = F.binary_cross_entropy_with_logits(logits_t0, occ_t0.view
        (batch_size, -1).to(device), reduction='none')
    loss_occ_t0 = loss_occ_t0.mean()
    return loss_occ_t0
