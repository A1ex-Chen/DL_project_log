@torch.no_grad()
def _match_and_label_boxes(self, proposals, stage, targets):
    """
        Match proposals with groundtruth using the matcher at the given stage.
        Label the proposals as foreground or background based on the match.

        Args:
            proposals (list[Instances]): One Instances for each image, with
                the field "proposal_boxes".
            stage (int): the current stage
            targets (list[Instances]): the ground truth instances

        Returns:
            list[Instances]: the same proposals, but with fields "gt_classes" and "gt_boxes"
        """
    num_fg_samples, num_bg_samples = [], []
    for proposals_per_image, targets_per_image in zip(proposals, targets):
        match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes,
            proposals_per_image.proposal_boxes)
        matched_idxs, proposal_labels = self.proposal_matchers[stage](
            match_quality_matrix)
        if len(targets_per_image) > 0:
            gt_classes = targets_per_image.gt_classes[matched_idxs]
            gt_classes[proposal_labels == 0] = self.num_classes
            gt_boxes = targets_per_image.gt_boxes[matched_idxs]
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
            gt_boxes = Boxes(targets_per_image.gt_boxes.tensor.new_zeros((
                len(proposals_per_image), 4)))
        proposals_per_image.gt_classes = gt_classes
        proposals_per_image.gt_boxes = gt_boxes
        num_fg_samples.append((proposal_labels == 1).sum().item())
        num_bg_samples.append(proposal_labels.numel() - num_fg_samples[-1])
    storage = get_event_storage()
    storage.put_scalar('stage{}/roi_head/num_fg_samples'.format(stage), sum
        (num_fg_samples) / len(num_fg_samples))
    storage.put_scalar('stage{}/roi_head/num_bg_samples'.format(stage), sum
        (num_bg_samples) / len(num_bg_samples))
    return proposals
