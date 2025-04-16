@torch.jit.unused
def losses(self, anchors: List[Boxes], pred_objectness_logits: List[torch.
    Tensor], gt_labels: List[torch.Tensor], pred_anchor_deltas: List[torch.
    Tensor], gt_boxes: List[torch.Tensor]) ->Dict[str, torch.Tensor]:
    """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A) representing
                the predicted objectness logits for all anchors.
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Tensor]): Output of :meth:`label_and_sample_anchors`.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
    num_images = len(gt_labels)
    gt_labels = torch.stack(gt_labels)
    pos_mask = gt_labels == 1
    num_pos_anchors = pos_mask.sum().item()
    num_neg_anchors = (gt_labels == 0).sum().item()
    storage = get_event_storage()
    storage.put_scalar('rpn/num_pos_anchors', num_pos_anchors / num_images)
    storage.put_scalar('rpn/num_neg_anchors', num_neg_anchors / num_images)
    localization_loss = _dense_box_regression_loss(anchors, self.
        box2box_transform, pred_anchor_deltas, gt_boxes, pos_mask,
        box_reg_loss_type=self.box_reg_loss_type, smooth_l1_beta=self.
        smooth_l1_beta)
    valid_mask = gt_labels >= 0
    objectness_loss = F.binary_cross_entropy_with_logits(cat(
        pred_objectness_logits, dim=1)[valid_mask], gt_labels[valid_mask].
        to(torch.float32), reduction='sum')
    normalizer = self.batch_size_per_image * num_images
    losses = {'loss_rpn_cls': objectness_loss / normalizer, 'loss_rpn_loc':
        localization_loss / normalizer}
    losses = {k: (v * self.loss_weight.get(k, 1.0)) for k, v in losses.items()}
    return losses
