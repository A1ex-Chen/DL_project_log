def call(self, inputs, img_info, gt_boxes=None, gt_labels=None, training=
    True, *args, **kwargs):
    cls_scores = self.conv_cls(inputs)
    bbox_preds = self.conv_reg(inputs)
    proposals = self.get_bboxes(cls_scores, bbox_preds, img_info, self.
        anchor_generator, gt_boxes=gt_boxes, gt_labels=gt_labels, training=
        training)
    return cls_scores, bbox_preds, proposals
