def _giou_loss(self, pred_deltas, anchors, gt_boxes):
    with autocast(False):
        pred_boxes = self.box2box_transform.apply_deltas(pred_deltas, anchors)
        loss = giou_loss(pred_boxes, gt_boxes, reduction='sum')
        return loss
