def compute_loss_corr(self, data, c_t=None, z_t=None):
    """ Returns the correspondence loss.

        Args:
            data (dict): data dictionary
            c_t (tensor): temporal conditioned code c_s
            z_t (tensor): latent temporal code z
        """
    if not self.loss_corr:
        return 0
    device = self.device
    pc = data.get('pointcloud').to(device)
    length_sequence = pc.shape[1]
    t = (torch.arange(length_sequence, dtype=torch.float32) / (
        length_sequence - 1)).to(device)
    if self.loss_corr_bw:
        pred_f = self.model.transform_to_t(t, pc[:, 0], c_t=c_t, z=z_t)
        pred_b = self.model.transform_to_t_backward(t, pc[:, -1], c_t=c_t,
            z=z_t)
        pred_b = pred_b.flip(1)
        lc1 = torch.norm(pred_f - pc, 2, dim=-1).mean()
        lc2 = torch.norm(pred_b - pc, 2, dim=-1).mean()
        loss_corr = lc1 + lc2
    else:
        pt_pred = self.model.transform_to_t(t[1:], pc[:, 0], c_t=c_t, z=z_t)
        loss_corr = torch.norm(pt_pred - pc[:, 1:], 2, dim=-1).mean()
    return loss_corr
