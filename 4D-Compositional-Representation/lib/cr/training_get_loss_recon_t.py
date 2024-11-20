def get_loss_recon_t(self, data, c_p=None, c_m=None, c_i=None, is_exchange=None
    ):
    """ Calculates the reconstruction loss.

        Args:
            data (tensor): training data
            c_t (tensor): temporal conditioned latent code
            z (tensor): latent code
        """
    device = self.device
    if is_exchange:
        p_t = data.get('points_t_ex').to(device)
        occ_t = data.get('points_t_ex.occ').to(device)
        time_val = data.get('points_t_ex.time').to(device)
    else:
        p_t = data.get('points_t').to(device)
        occ_t = data.get('points_t.occ').to(device)
        time_val = data.get('points_t.time').to(device)
    batch_size, n_pts, _ = p_t.shape
    c_p_at_t = self.model.transform_to_t(time_val, p=c_p, c_t=c_m)
    c_s_at_t = torch.cat([c_i, c_p_at_t], 1)
    p = p_t
    logits_pred = self.model.decode(p, c=c_s_at_t).logits
    loss_occ_t = F.binary_cross_entropy_with_logits(logits_pred, occ_t.view
        (batch_size, -1), reduction='none')
    loss_occ_t = loss_occ_t.mean()
    return loss_occ_t
