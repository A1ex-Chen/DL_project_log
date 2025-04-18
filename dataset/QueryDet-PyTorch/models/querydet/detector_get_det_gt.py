@torch.no_grad()
def get_det_gt(self, anchors, targets):
    gt_classes = []
    gt_anchors_targets = []
    anchor_layers = len(anchors)
    anchor_lens = [len(x) for x in anchors]
    start_inds = [0] + [sum(anchor_lens[:i]) for i in range(1, len(
        anchor_lens))]
    end_inds = [sum(anchor_lens[:i + 1]) for i in range(len(anchor_lens))]
    all_anchors = Boxes.cat(anchors)
    for targets_per_image in targets:
        if type(self.matcher) == Matcher:
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes,
                all_anchors)
            gt_matched_idxs, anchor_labels = self.matcher(match_quality_matrix)
            del match_quality_matrix
        elif type(self.matcher) == LoopMatcher:
            gt_matched_idxs, anchor_labels = self.matcher(targets_per_image
                .gt_boxes, all_anchors)
        else:
            raise NotImplementedError
        has_gt = len(targets_per_image) > 0
        if has_gt:
            matched_gt_boxes = targets_per_image.gt_boxes[gt_matched_idxs]
            if not self.use_giou_loss:
                gt_anchors_reg_targets_i = self.box2box_transform.get_deltas(
                    all_anchors.tensor, matched_gt_boxes.tensor)
            else:
                gt_anchors_reg_targets_i = matched_gt_boxes.tensor
            gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
            gt_classes_i[anchor_labels == 0] = self.num_classes
            gt_classes_i[anchor_labels == -1] = -1
        else:
            gt_classes_i = torch.zeros_like(gt_matched_idxs) + self.num_classes
            gt_anchors_reg_targets_i = torch.zeros_like(all_anchors.tensor)
        gt_classes.append([gt_classes_i[s:e] for s, e in zip(start_inds,
            end_inds)])
        gt_anchors_targets.append([gt_anchors_reg_targets_i[s:e] for s, e in
            zip(start_inds, end_inds)])
    gt_classes = [torch.stack([x[i] for x in gt_classes]) for i in range(
        anchor_layers)]
    gt_anchors_targets = [torch.stack([x[i] for x in gt_anchors_targets]) for
        i in range(anchor_layers)]
    gt_classes = torch.cat([x.flatten() for x in gt_classes])
    gt_anchors_targets = torch.cat([x.reshape(-1, 4) for x in
        gt_anchors_targets])
    return gt_classes, gt_anchors_targets
