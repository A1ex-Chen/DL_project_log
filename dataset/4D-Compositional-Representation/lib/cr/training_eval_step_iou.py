def eval_step_iou(self, data, c_p=None, c_m=None, c_i=None):
    """ Calculates the IoU for the evaluation step.

        Args:
            data (tensor): training data
            c_t (tensor): temporal conditioned latent code
            z (tensor): latent code
        """
    device = self.device
    threshold = self.threshold
    eval_dict = {}
    pts_iou = data.get('points_iou').to(device)
    occ_iou = data.get('points_iou.occ')
    pts_iou_t = data.get('points_iou.time').to(device)
    batch_size, n_steps, n_pts, dim = pts_iou.shape
    p = pts_iou
    c_i = c_i.unsqueeze(0).repeat(1, n_steps, 1)
    c_p_at_t = self.model.transform_to_t_eval(pts_iou_t[0], p=c_p, c_t=c_m)
    c_s_at_t = torch.cat([c_i, c_p_at_t], -1)
    c_s_at_t = c_s_at_t.view(batch_size * n_steps, c_s_at_t.shape[-1])
    p = p.view(batch_size * n_steps, n_pts, -1)
    occ_iou = occ_iou.view(batch_size * n_steps, n_pts)
    occ_pred = self.model.decode(p, c=c_s_at_t)
    occ_pred = (occ_pred.probs > threshold).cpu().numpy()
    occ_gt = (occ_iou >= 0.5).numpy()
    iou = compute_iou(occ_pred, occ_gt)
    iou = iou.reshape(batch_size, -1).mean(0)
    eval_dict['iou'] = iou.sum() / len(iou)
    for i in range(len(iou)):
        eval_dict['iou_t%d' % i] = iou[i]
    return eval_dict
