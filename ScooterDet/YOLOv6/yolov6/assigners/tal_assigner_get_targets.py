def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
    batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=
        gt_labels.device)[..., None]
    target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
    target_labels = gt_labels.long().flatten()[target_gt_idx]
    target_bboxes = gt_bboxes.reshape([-1, 4])[target_gt_idx]
    target_labels[target_labels < 0] = 0
    target_scores = F.one_hot(target_labels, self.num_classes)
    fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
    target_scores = torch.where(fg_scores_mask > 0, target_scores, torch.
        full_like(target_scores, 0))
    return target_labels, target_bboxes, target_scores
