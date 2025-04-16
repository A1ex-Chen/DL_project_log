def add_ground_truth_to_proposals(gt: Union[List[Instances], List[Boxes]],
    proposals: List[Instances]) ->List[Instances]:
    """
    Call `add_ground_truth_to_proposals_single_image` for all images.

    Args:
        gt(Union[List[Instances], List[Boxes]): list of N elements. Element i is a Instances
            representing the ground-truth for image i.
        proposals (list[Instances]): list of N elements. Element i is a Instances
            representing the proposals for image i.

    Returns:
        list[Instances]: list of N Instances. Each is the proposals for the image,
            with field "proposal_boxes" and "objectness_logits".
    """
    assert gt is not None
    if len(proposals) != len(gt):
        raise ValueError(
            'proposals and gt should have the same length as the number of images!'
            )
    if len(proposals) == 0:
        return proposals
    return [add_ground_truth_to_proposals_single_image(gt_i, proposals_i) for
        gt_i, proposals_i in zip(gt, proposals)]
