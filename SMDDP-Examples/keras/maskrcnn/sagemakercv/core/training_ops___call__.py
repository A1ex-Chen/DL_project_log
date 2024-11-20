def __call__(self, boxes, gt_boxes, gt_labels):
    (sample_box_targets, class_targets, rois, sample_proposal_to_label_map) = (
        proposal_label_op(boxes, gt_boxes, gt_labels, batch_size_per_im=
        self.batch_size_per_im, fg_fraction=self.fg_fraction, fg_thresh=
        self.fg_thresh, bg_thresh_hi=self.bg_thresh_hi, bg_thresh_lo=self.
        bg_thresh_lo))
    return (sample_box_targets, class_targets, rois,
        sample_proposal_to_label_map)
