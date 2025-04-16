def _process_unmatched_prev_idx(self, instances: Instances,
    matched_prev_idx: np.ndarray) ->Instances:
    untracked_instances = Instances(image_size=instances.image_size,
        pred_boxes=[], pred_masks=[], pred_classes=[], scores=[], ID=[],
        ID_period=[], lost_frame_count=[])
    prev_bboxes = list(self._prev_instances.pred_boxes)
    prev_classes = list(self._prev_instances.pred_classes)
    prev_scores = list(self._prev_instances.scores)
    prev_ID_period = self._prev_instances.ID_period
    if instances.has('pred_masks'):
        prev_masks = list(self._prev_instances.pred_masks)
    untracked_prev_idx = set(range(len(self._prev_instances))).difference(set
        (matched_prev_idx))
    for idx in untracked_prev_idx:
        x_left, y_top, x_right, y_bot = prev_bboxes[idx]
        if (1.0 * (x_right - x_left) / self._video_width < self.
            _min_box_rel_dim or 1.0 * (y_bot - y_top) / self._video_height <
            self._min_box_rel_dim or self._prev_instances.lost_frame_count[
            idx] >= self._max_lost_frame_count or prev_ID_period[idx] <=
            self._min_instance_period):
            continue
        untracked_instances.pred_boxes.append(list(prev_bboxes[idx].numpy()))
        untracked_instances.pred_classes.append(int(prev_classes[idx]))
        untracked_instances.scores.append(float(prev_scores[idx]))
        untracked_instances.ID.append(self._prev_instances.ID[idx])
        untracked_instances.ID_period.append(self._prev_instances.ID_period
            [idx])
        untracked_instances.lost_frame_count.append(self._prev_instances.
            lost_frame_count[idx] + 1)
        if instances.has('pred_masks'):
            untracked_instances.pred_masks.append(prev_masks[idx].numpy().
                astype(np.uint8))
    untracked_instances.pred_boxes = Boxes(torch.FloatTensor(
        untracked_instances.pred_boxes))
    untracked_instances.pred_classes = torch.IntTensor(untracked_instances.
        pred_classes)
    untracked_instances.scores = torch.FloatTensor(untracked_instances.scores)
    if instances.has('pred_masks'):
        untracked_instances.pred_masks = torch.IntTensor(untracked_instances
            .pred_masks)
    else:
        untracked_instances.remove('pred_masks')
    return Instances.cat([instances, untracked_instances])
