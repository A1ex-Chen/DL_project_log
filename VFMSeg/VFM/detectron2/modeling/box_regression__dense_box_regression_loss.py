def _dense_box_regression_loss(anchors: List[Union[Boxes, torch.Tensor]],
    box2box_transform: Box2BoxTransform, pred_anchor_deltas: List[torch.
    Tensor], gt_boxes: List[torch.Tensor], fg_mask: torch.Tensor,
    box_reg_loss_type='smooth_l1', smooth_l1_beta=0.0):
    """
    Compute loss for dense multi-level box regression.
    Loss is accumulated over ``fg_mask``.

    Args:
        anchors: #lvl anchor boxes, each is (HixWixA, 4)
        pred_anchor_deltas: #lvl predictions, each is (N, HixWixA, 4)
        gt_boxes: N ground truth boxes, each has shape (R, 4) (R = sum(Hi * Wi * A))
        fg_mask: the foreground boolean mask of shape (N, R) to compute loss on
        box_reg_loss_type (str): Loss type to use. Supported losses: "smooth_l1", "giou",
            "diou", "ciou".
        smooth_l1_beta (float): beta parameter for the smooth L1 regression loss. Default to
            use L1 loss. Only used when `box_reg_loss_type` is "smooth_l1"
    """
    if isinstance(anchors[0], Boxes):
        anchors = type(anchors[0]).cat(anchors).tensor
    else:
        anchors = cat(anchors)
    if box_reg_loss_type == 'smooth_l1':
        gt_anchor_deltas = [box2box_transform.get_deltas(anchors, k) for k in
            gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)
        loss_box_reg = smooth_l1_loss(cat(pred_anchor_deltas, dim=1)[
            fg_mask], gt_anchor_deltas[fg_mask], beta=smooth_l1_beta,
            reduction='sum')
    elif box_reg_loss_type == 'giou':
        pred_boxes = [box2box_transform.apply_deltas(k, anchors) for k in
            cat(pred_anchor_deltas, dim=1)]
        loss_box_reg = giou_loss(torch.stack(pred_boxes)[fg_mask], torch.
            stack(gt_boxes)[fg_mask], reduction='sum')
    elif box_reg_loss_type == 'diou':
        pred_boxes = [box2box_transform.apply_deltas(k, anchors) for k in
            cat(pred_anchor_deltas, dim=1)]
        loss_box_reg = diou_loss(torch.stack(pred_boxes)[fg_mask], torch.
            stack(gt_boxes)[fg_mask], reduction='sum')
    elif box_reg_loss_type == 'ciou':
        pred_boxes = [box2box_transform.apply_deltas(k, anchors) for k in
            cat(pred_anchor_deltas, dim=1)]
        loss_box_reg = ciou_loss(torch.stack(pred_boxes)[fg_mask], torch.
            stack(gt_boxes)[fg_mask], reduction='sum')
    else:
        raise ValueError(
            f"Invalid dense box regression loss type '{box_reg_loss_type}'")
    return loss_box_reg
