def get_bboxes(self, cls_scores, bbox_preds, img_info, anchors, training=True):
    rpn_box_rois, rpn_box_scores = self.roi_proposal(cls_scores, bbox_preds,
        img_info, anchors, training=training)
    return rpn_box_rois, rpn_box_scores
