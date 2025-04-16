def losses(self, anchors, pred_logits, gt_labels, pred_anchor_deltas,
    gt_boxes, pred_centerness):
    """
        This method is almost identical to :meth:`RetinaNet.losses`, with an extra
        "loss_centerness" in the returned dict.
        """
    num_images = len(gt_labels)
    gt_labels = torch.stack(gt_labels)
    pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
    num_pos_anchors = pos_mask.sum().item()
    get_event_storage().put_scalar('num_pos_anchors', num_pos_anchors /
        num_images)
    normalizer = self._ema_update('loss_normalizer', max(num_pos_anchors, 1
        ), 300)
    gt_labels_target = F.one_hot(gt_labels, num_classes=self.num_classes + 1)[
        :, :, :-1]
    loss_cls = sigmoid_focal_loss_jit(torch.cat(pred_logits, dim=1),
        gt_labels_target.to(pred_logits[0].dtype), alpha=self.
        focal_loss_alpha, gamma=self.focal_loss_gamma, reduction='sum')
    loss_box_reg = _dense_box_regression_loss(anchors, self.
        box2box_transform, pred_anchor_deltas, gt_boxes, pos_mask,
        box_reg_loss_type='giou')
    ctrness_targets = self.compute_ctrness_targets(anchors, gt_boxes)
    pred_centerness = torch.cat(pred_centerness, dim=1).squeeze(dim=2)
    ctrness_loss = F.binary_cross_entropy_with_logits(pred_centerness[
        pos_mask], ctrness_targets[pos_mask], reduction='sum')
    return {'loss_fcos_cls': loss_cls / normalizer, 'loss_fcos_loc': 
        loss_box_reg / normalizer, 'loss_fcos_ctr': ctrness_loss / normalizer}
