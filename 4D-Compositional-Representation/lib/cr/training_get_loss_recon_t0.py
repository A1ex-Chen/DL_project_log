def get_loss_recon_t0(self, data, c_p=None, c_i=None, is_exchange=None):
    """ Calculates the reconstruction loss.

        Args:
            data (tensor): training data
            c_t (tensor): temporal conditioned latent code
            z (tensor): latent code
        """
    if is_exchange:
        p_t0 = data.get('points_ex')
        occ_t0 = data.get('points_ex.occ')
    else:
        p_t0 = data.get('points')
        occ_t0 = data.get('points.occ')
    batch_size, n_pts, _ = p_t0.shape
    device = self.device
    batch_size = p_t0.shape[0]
    c_s_at_t0 = torch.cat([c_i, c_p], 1)
    p = p_t0
    logits_t0 = self.model.decode(p.to(device), c=c_s_at_t0).logits
    loss_occ_t0 = F.binary_cross_entropy_with_logits(logits_t0, occ_t0.view
        (batch_size, -1).to(device), reduction='none')
    loss_occ_t0 = loss_occ_t0.mean()
    return loss_occ_t0
