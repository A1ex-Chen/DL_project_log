def compute_loss(self, data):
    """ Calculates the loss.

        Args:
            data (tensor): training data
        """
    device = self.device
    seq1, seq2 = data
    inputs1 = seq1.get('inputs', torch.empty(1, 1, 0)).to(device)
    inputs2 = seq2.get('inputs', torch.empty(1, 1, 0)).to(device)
    c_p_1, c_m_1, c_i_1 = self.model.encode_inputs(inputs1)
    c_p_2, c_m_2, c_i_2 = self.model.encode_inputs(inputs2)
    is_exchange = np.random.randint(2)
    if is_exchange:
        in_c_i_1 = c_i_2
        in_c_i_2 = c_i_1
    else:
        in_c_i_1 = c_i_1
        in_c_i_2 = c_i_2
    loss_recon_t_1 = self.get_loss_recon_t(seq1, c_m=c_m_1, c_p=c_p_1, c_i=
        in_c_i_1, is_exchange=is_exchange)
    loss_recon_t0_1 = self.get_loss_recon_t0(seq1, c_p=c_p_1, c_i=in_c_i_1,
        is_exchange=is_exchange)
    loss_recon_t_2 = self.get_loss_recon_t(seq2, c_m=c_m_2, c_p=c_p_2, c_i=
        in_c_i_2, is_exchange=is_exchange)
    loss_recon_t0_2 = self.get_loss_recon_t0(seq2, c_p=c_p_2, c_i=in_c_i_2,
        is_exchange=is_exchange)
    loss_recon_t = (loss_recon_t_1 + loss_recon_t_2) / 2.0
    loss_recon_t0 = (loss_recon_t0_1 + loss_recon_t0_2) / 2.0
    loss = loss_recon_t + loss_recon_t0
    return loss
