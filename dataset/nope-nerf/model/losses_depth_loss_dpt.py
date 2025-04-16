def depth_loss_dpt(self, pred_depth, gt_depth, weight=None):
    """
        :param pred_depth:  (H, W)
        :param gt_depth:    (H, W)
        :param weight:      (H, W)
        :return:            scalar
        """
    t_pred = torch.median(pred_depth)
    s_pred = torch.mean(torch.abs(pred_depth - t_pred))
    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))
    pred_depth_n = (pred_depth - t_pred) / s_pred
    gt_depth_n = (gt_depth - t_gt) / s_gt
    if weight is not None:
        loss = F.mse_loss(pred_depth_n, gt_depth_n, reduction='none')
        loss = loss * weight
        loss = loss.sum() / (weight.sum() + 1e-08)
    else:
        loss = F.mse_loss(pred_depth_n, gt_depth_n)
    return loss
