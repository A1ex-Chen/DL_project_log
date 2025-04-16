def add_ground_truth_to_proposals_single_image(gt: Union[Instances, Boxes],
    proposals: Instances) ->Instances:
    """
    Augment `proposals` with `gt`.

    Args:
        Same as `add_ground_truth_to_proposals`, but with gt and proposals
        per image.

    Returns:
        Same as `add_ground_truth_to_proposals`, but for only one image.
    """
    if isinstance(gt, Boxes):
        gt = Instances(proposals.image_size, gt_boxes=gt)
    gt_boxes = gt.gt_boxes
    device = proposals.objectness_logits.device
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
    gt_logits = gt_logit_value * torch.ones(len(gt_boxes), device=device)
    gt_proposal = Instances(proposals.image_size, **gt.get_fields())
    gt_proposal.proposal_boxes = gt_boxes
    gt_proposal.objectness_logits = gt_logits
    for key in proposals.get_fields().keys():
        assert gt_proposal.has(key
            ), "The attribute '{}' in `proposals` does not exist in `gt`".format(
            key)
    new_proposals = Instances.cat([proposals, gt_proposal])
    return new_proposals
