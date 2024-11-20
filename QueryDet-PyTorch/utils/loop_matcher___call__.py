def __call__(self, gt_boxes, anchors):
    if len(gt_boxes) == 0:
        default_matches = torch.zeros(len(anchors), dtype=torch.int64).to(
            anchors.tensor.device)
        default_match_labels = torch.zeros(len(anchors), dtype=torch.int8).to(
            anchors.tensor.device) + self.labels[0]
        return default_matches, default_match_labels
    gt_boxes_tensor = gt_boxes.tensor
    anchors_tensor = anchors.tensor
    max_ious = torch.zeros(len(anchors)).to(anchors_tensor.device)
    matched_inds = torch.zeros(len(anchors), dtype=torch.long).to(
        anchors_tensor.device)
    gt_ious = torch.zeros(len(gt_boxes)).to(anchors_tensor.device)
    for i in range(len(gt_boxes)):
        ious = self._iou(anchors_tensor, gt_boxes_tensor[i])
        gt_ious[i] = ious.max()
        matched_inds = torch.where(ious > max_ious, torch.zeros(1, dtype=
            torch.long, device=matched_inds.device) + i, matched_inds)
        max_ious = torch.max(ious, max_ious)
        del ious
    matched_vals = max_ious
    matches = matched_inds
    match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)
    for l, low, high in zip(self.labels, self.thresholds[:-1], self.
        thresholds[1:]):
        low_high = (matched_vals >= low) & (matched_vals < high)
        match_labels[low_high] = l
    if self.allow_low_quality_matches:
        self.set_low_quality_matches_(match_labels, matched_vals, matches,
            gt_ious)
    return matches, match_labels
