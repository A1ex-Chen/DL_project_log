def eval_step_corr_l2(self, data, c_t=None, z_t=None):
    """ Calculates the correspondence l2 distance for an evaluation test set item.

        Args:
            data (dict): data dictionary
            c_s (tensor): spatial conditioned code
            c_t (tensor): temporal conditioned code
            z (tensor): latent code
        """
    eval_dict = {}
    device = self.device
    p_mesh = data.get('points_mesh').to(device)
    p_mesh_t = data.get('points_mesh.time').to(device)[0]
    n_steps = p_mesh_t.shape[0]
    pts_pred = self.model.transform_to_t(p_mesh_t, p_mesh[:, 0], z_t, c_t)
    if self.loss_corr_bw:
        pred_b = self.model.transform_to_t_backward(p_mesh_t, p_mesh[:, -1],
            z_t, c_t).flip(1)
        w = (torch.arange(n_steps).float() / (n_steps - 1)).view(1, n_steps,
            1, 1).to(device)
        pts_pred = pts_pred * (1 - w) + pred_b * w
    l2 = torch.norm(pts_pred - p_mesh, 2, dim=-1).mean(0).mean(-1)
    eval_dict['l2'] = l2.sum().item() / len(l2)
    for i in range(len(l2)):
        eval_dict['l2_%d' % (i + 1)] = l2[i].item()
    return eval_dict
