def update(self, instances: Instances) ->Instances:
    """
        See BaseTracker description
        """
    if instances.has('pred_keypoints'):
        raise NotImplementedError('Need to add support for keypoints')
    instances = self._initialize_extra_fields(instances)
    if self._prev_instances is not None:
        iou_all = pairwise_iou(boxes1=instances.pred_boxes, boxes2=self.
            _prev_instances.pred_boxes)
        bbox_pairs = self._create_prediction_pairs(instances, iou_all)
        self._reset_fields()
        for bbox_pair in bbox_pairs:
            idx = bbox_pair['idx']
            prev_id = bbox_pair['prev_id']
            if (idx in self._matched_idx or prev_id in self._matched_ID or 
                bbox_pair['IoU'] < self._track_iou_threshold):
                continue
            instances.ID[idx] = prev_id
            instances.ID_period[idx] = bbox_pair['prev_period'] + 1
            instances.lost_frame_count[idx] = 0
            self._matched_idx.add(idx)
            self._matched_ID.add(prev_id)
            self._untracked_prev_idx.remove(bbox_pair['prev_idx'])
        instances = self._assign_new_id(instances)
        instances = self._merge_untracked_instances(instances)
    self._prev_instances = copy.deepcopy(instances)
    return instances
