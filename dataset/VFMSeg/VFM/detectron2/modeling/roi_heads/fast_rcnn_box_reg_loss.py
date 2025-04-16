def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
    """
        Args:
            proposal_boxes/gt_boxes are tensors with the same shape (R, 4 or 5).
            pred_deltas has shape (R, 4 or 5), or (R, num_classes * (4 or 5)).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
    box_dim = proposal_boxes.shape[1]
    fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes)
        )[0]
    if pred_deltas.shape[1] == box_dim:
        fg_pred_deltas = pred_deltas[fg_inds]
    else:
        fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
            fg_inds, gt_classes[fg_inds]]
    loss_box_reg = _dense_box_regression_loss([proposal_boxes[fg_inds]],
        self.box2box_transform, [fg_pred_deltas.unsqueeze(0)], [gt_boxes[
        fg_inds]], ..., self.box_reg_loss_type, self.smooth_l1_beta)
    return loss_box_reg / max(gt_classes.numel(), 1.0)
