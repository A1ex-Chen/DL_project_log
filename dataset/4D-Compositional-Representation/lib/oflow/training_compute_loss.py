def compute_loss(self, data):
    """ Computes the loss.

        Args:
            data (dict): data dictionary
        """
    device = self.device
    inputs = data.get('inputs', torch.empty(1, 1, 0)).to(device)
    c_s, c_t = self.model.encode_inputs(inputs)
    q_z, q_z_t = self.model.infer_z(inputs, c=c_t, data=data)
    z, z_t = q_z.rsample(), q_z_t.rsample()
    loss_kl = self.compute_kl(q_z) + self.compute_kl(q_z_t)
    loss_recon = self.get_loss_recon(data, c_s, c_t, z, z_t)
    loss_corr = self.compute_loss_corr(data, c_t, z_t)
    loss = loss_recon + loss_corr + loss_kl
    return loss
