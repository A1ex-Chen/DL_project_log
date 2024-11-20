@torch.no_grad()
def label_and_sample_proposals(self, proposals, targets):
    """
        Prepare some proposals to be used to train the RROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
        with a fraction of positives that is no larger than `self.positive_sample_fraction.

        Args:
            See :meth:`StandardROIHeads.forward`

        Returns:
            list[Instances]: length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the rotated proposal boxes
                - gt_boxes: the ground-truth rotated boxes that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                - gt_classes: the ground-truth classification lable for each proposal
        """
    if self.proposal_append_gt:
        proposals = add_ground_truth_to_proposals(targets, proposals)
    proposals_with_gt = []
    num_fg_samples = []
    num_bg_samples = []
    for proposals_per_image, targets_per_image in zip(proposals, targets):
        has_gt = len(targets_per_image) > 0
        match_quality_matrix = pairwise_iou_rotated(targets_per_image.
            gt_boxes, proposals_per_image.proposal_boxes)
        matched_idxs, matched_labels = self.proposal_matcher(
            match_quality_matrix)
        sampled_idxs, gt_classes = self._sample_proposals(matched_idxs,
            matched_labels, targets_per_image.gt_classes)
        proposals_per_image = proposals_per_image[sampled_idxs]
        proposals_per_image.gt_classes = gt_classes
        if has_gt:
            sampled_targets = matched_idxs[sampled_idxs]
            proposals_per_image.gt_boxes = targets_per_image.gt_boxes[
                sampled_targets]
        num_bg_samples.append((gt_classes == self.num_classes).sum().item())
        num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
        proposals_with_gt.append(proposals_per_image)
    storage = get_event_storage()
    storage.put_scalar('roi_head/num_fg_samples', np.mean(num_fg_samples))
    storage.put_scalar('roi_head/num_bg_samples', np.mean(num_bg_samples))
    return proposals_with_gt
