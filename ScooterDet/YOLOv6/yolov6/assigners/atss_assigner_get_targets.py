def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
    batch_idx = torch.arange(self.bs, dtype=gt_labels.dtype, device=
        gt_labels.device)
    batch_idx = batch_idx[..., None]
    target_gt_idx = (target_gt_idx + batch_idx * self.n_max_boxes).long()
    target_labels = gt_labels.flatten()[target_gt_idx.flatten()]
    target_labels = target_labels.reshape([self.bs, self.n_anchors])
    target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like
        (target_labels, self.bg_idx))
    target_bboxes = gt_bboxes.reshape([-1, 4])[target_gt_idx.flatten()]
    target_bboxes = target_bboxes.reshape([self.bs, self.n_anchors, 4])
    target_scores = F.one_hot(target_labels.long(), self.num_classes + 1
        ).float()
    target_scores = target_scores[:, :, :self.num_classes]
    return target_labels, target_bboxes, target_scores
