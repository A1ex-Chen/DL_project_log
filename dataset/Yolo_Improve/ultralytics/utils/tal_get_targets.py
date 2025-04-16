def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
    """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                          positive anchor points.
                - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                          for positive anchor points.
                - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                          for positive anchor points, where num_classes is the number
                                          of object classes.
        """
    batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=
        gt_labels.device)[..., None]
    target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
    target_labels = gt_labels.long().flatten()[target_gt_idx]
    target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]
    target_labels.clamp_(0)
    target_scores = torch.zeros((target_labels.shape[0], target_labels.
        shape[1], self.num_classes), dtype=torch.int64, device=
        target_labels.device)
    target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
    fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
    target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)
    return target_labels, target_bboxes, target_scores
