def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=''
    ):
    """Computes the classification loss based on predictions, target values, and ground truth scores."""
    name_class = f'loss_class{postfix}'
    bs, nq = pred_scores.shape[:2]
    one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=
        targets.device)
    one_hot.scatter_(2, targets.unsqueeze(-1), 1)
    one_hot = one_hot[..., :-1]
    gt_scores = gt_scores.view(bs, nq, 1) * one_hot
    if self.fl:
        if num_gts and self.vfl:
            loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
        else:
            loss_cls = self.fl(pred_scores, one_hot.float())
        loss_cls /= max(num_gts, 1) / nq
    else:
        loss_cls = nn.BCEWithLogitsLoss(reduction='none')(pred_scores,
            gt_scores).mean(1).sum()
    return {name_class: loss_cls.squeeze() * self.loss_gain['class']}
