def forward(self, pred_dist, pred_bboxes, t_pred_dist, t_pred_bboxes,
    temperature, anchor_points, target_bboxes, target_scores,
    target_scores_sum, fg_mask):
    num_pos = fg_mask.sum()
    if num_pos > 0:
        bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
        pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).reshape([
            -1, 4])
        t_pred_bboxes_pos = torch.masked_select(t_pred_bboxes, bbox_mask
            ).reshape([-1, 4])
        target_bboxes_pos = torch.masked_select(target_bboxes, bbox_mask
            ).reshape([-1, 4])
        bbox_weight = torch.masked_select(target_scores.sum(-1), fg_mask
            ).unsqueeze(-1)
        loss_iou = self.iou_loss(pred_bboxes_pos, target_bboxes_pos
            ) * bbox_weight
        if target_scores_sum == 0:
            loss_iou = loss_iou.sum()
        else:
            loss_iou = loss_iou.sum() / target_scores_sum
        if self.use_dfl:
            dist_mask = fg_mask.unsqueeze(-1).repeat([1, 1, (self.reg_max +
                1) * 4])
            pred_dist_pos = torch.masked_select(pred_dist, dist_mask).reshape([
                -1, 4, self.reg_max + 1])
            t_pred_dist_pos = torch.masked_select(t_pred_dist, dist_mask
                ).reshape([-1, 4, self.reg_max + 1])
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            target_ltrb_pos = torch.masked_select(target_ltrb, bbox_mask
                ).reshape([-1, 4])
            loss_dfl = self._df_loss(pred_dist_pos, target_ltrb_pos
                ) * bbox_weight
            d_loss_dfl = self.distill_loss_dfl(pred_dist_pos,
                t_pred_dist_pos, temperature) * bbox_weight
            if target_scores_sum == 0:
                loss_dfl = loss_dfl.sum()
                d_loss_dfl = d_loss_dfl.sum()
            else:
                loss_dfl = loss_dfl.sum() / target_scores_sum
                d_loss_dfl = d_loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = pred_dist.sum() * 0.0
            d_loss_dfl = pred_dist.sum() * 0.0
    else:
        loss_iou = pred_dist.sum() * 0.0
        loss_dfl = pred_dist.sum() * 0.0
        d_loss_dfl = pred_dist.sum() * 0.0
    return loss_iou, loss_dfl, d_loss_dfl
