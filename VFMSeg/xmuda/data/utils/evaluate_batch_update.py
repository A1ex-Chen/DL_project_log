def batch_update(self, pred_labels, gt_labels):
    assert len(pred_labels) == len(gt_labels)
    for pred_label, gt_label in zip(pred_labels, gt_labels):
        self.update(pred_label, gt_label)
