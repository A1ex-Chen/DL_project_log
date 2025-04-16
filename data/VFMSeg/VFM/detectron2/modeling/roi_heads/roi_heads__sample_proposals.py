def _sample_proposals(self, matched_idxs: torch.Tensor, matched_labels:
    torch.Tensor, gt_classes: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor
    ]:
    """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
    has_gt = gt_classes.numel() > 0
    if has_gt:
        gt_classes = gt_classes[matched_idxs]
        gt_classes[matched_labels == 0] = self.num_classes
        gt_classes[matched_labels == -1] = -1
    else:
        gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
    sampled_fg_idxs, sampled_bg_idxs = subsample_labels(gt_classes, self.
        batch_size_per_image, self.positive_fraction, self.num_classes)
    sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
    return sampled_idxs, gt_classes[sampled_idxs]
