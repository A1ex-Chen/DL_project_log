def eval_step_iou(self, data, c_s=None, c_t=None, z=None, z_t=None):
    """ Calculates the IoU score for an evaluation test set item.

        Args:
            data (dict): data dictionary
            c_s (tensor): spatial conditioned code
            c_t (tensor): temporal conditioned code
            z (tensor): latent shape code
            z_t (tensor): latent motion code
        """
    device = self.device
    threshold = self.threshold
    eval_dict = {}
    pts_iou = data.get('points').to(device)
    occ_iou = data.get('points.occ').squeeze(0)
    pts_iou_t = data.get('points.time').to(device)
    batch_size, n_steps, n_pts, dim = pts_iou.shape
    pts_iou_t0 = torch.stack([self.model.transform_to_t0(pts_iou_t[:, i],
        pts_iou[:, i], z_t, c_t) for i in range(n_steps)], dim=1)
    c_s = c_s.unsqueeze(1).repeat(1, n_steps, 1).view(batch_size * n_steps, -1)
    z = z.unsqueeze(1).repeat(1, n_steps, 1).view(batch_size * n_steps, -1)
    pts_iou_t0 = pts_iou_t0.view(batch_size * n_steps, n_pts, dim)
    p_r = self.model.decode(pts_iou_t0, z, c_s)
    rec_error = -p_r.log_prob(occ_iou.to(device).view(-1, n_pts)).mean(-1)
    rec_error = rec_error.view(batch_size, n_steps).mean(0)
    occ_pred = (p_r.probs > threshold).view(batch_size, n_steps, n_pts).cpu(
        ).numpy()
    occ_gt = (occ_iou >= 0.5).numpy()
    iou = compute_iou(occ_pred.reshape(-1, n_pts), occ_gt.reshape(-1, n_pts))
    iou = iou.reshape(batch_size, n_steps).mean(0)
    eval_dict['iou'] = iou.sum() / len(iou)
    eval_dict['rec_error'] = rec_error.sum().item() / len(rec_error)
    for i in range(len(iou)):
        eval_dict['iou_t%d' % i] = iou[i]
        eval_dict['rec_error_t%d' % i] = rec_error[i].item()
    return eval_dict
