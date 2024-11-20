def get_loss_recon_t(self, data, c_s=None, c_t=None, z=None, z_t=None):
    """ Returns the reconstruction loss for time step t>0.

        Args:
            data (dict): data dictionary
            c_s (tensor): spatial conditioned code c_s
            c_t (tensor): temporal conditioned code c_s
            z (tensor): latent code z
            z_t (tensor): latent temporal code z
        """
    device = self.device
    p_t = data.get('points_t').to(device)
    occ_t = data.get('points_t.occ').to(device)
    time_val = data.get('points_t.time').to(device)
    batch_size, n_pts, p_dim = p_t.shape
    p_t_at_t0 = self.model.transform_to_t0(time_val, p_t, c_t=c_t, z=z_t)
    logits_p_t = self.model.decode(p_t_at_t0, c=c_s, z=z).logits
    loss_occ_t = F.binary_cross_entropy_with_logits(logits_p_t, occ_t.view(
        batch_size, -1), reduction='none')
    loss_occ_t = loss_occ_t.mean()
    return loss_occ_t
