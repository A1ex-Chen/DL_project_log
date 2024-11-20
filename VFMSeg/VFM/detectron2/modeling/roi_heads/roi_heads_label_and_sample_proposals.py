@torch.no_grad()
def label_and_sample_proposals(self, proposals: List[Instances], targets:
    List[Instances]) ->List[Instances]:
    """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
    if self.proposal_append_gt:
        proposals = add_ground_truth_to_proposals(targets, proposals)
    proposals_with_gt = []
    num_fg_samples = []
    num_bg_samples = []
    for proposals_per_image, targets_per_image in zip(proposals, targets):
        has_gt = len(targets_per_image) > 0
        match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes,
            proposals_per_image.proposal_boxes)
        matched_idxs, matched_labels = self.proposal_matcher(
            match_quality_matrix)
        sampled_idxs, gt_classes = self._sample_proposals(matched_idxs,
            matched_labels, targets_per_image.gt_classes)
        proposals_per_image = proposals_per_image[sampled_idxs]
        proposals_per_image.gt_classes = gt_classes
        if has_gt:
            sampled_targets = matched_idxs[sampled_idxs]
            for trg_name, trg_value in targets_per_image.get_fields().items():
                if trg_name.startswith('gt_') and not proposals_per_image.has(
                    trg_name):
                    proposals_per_image.set(trg_name, trg_value[
                        sampled_targets])
        num_bg_samples.append((gt_classes == self.num_classes).sum().item())
        num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
        proposals_with_gt.append(proposals_per_image)
    storage = get_event_storage()
    storage.put_scalar('roi_head/num_fg_samples', np.mean(num_fg_samples))
    storage.put_scalar('roi_head/num_bg_samples', np.mean(num_bg_samples))
    return proposals_with_gt
