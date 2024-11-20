def forward_training(self, images, features, predictions, gt_instances):
    pred_logits, pred_anchor_deltas, pred_centerness = (self.
        _transpose_dense_predictions(predictions, [self.num_classes, 4, 1]))
    anchors = self.anchor_generator(features)
    gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
    return self.losses(anchors, pred_logits, gt_labels, pred_anchor_deltas,
        gt_boxes, pred_centerness)
