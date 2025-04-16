def _create_prediction_pairs(self, instances: Instances, iou_all: np.ndarray
    ) ->List:
    """
        For all instances in previous and current frames, create pairs. For each
        pair, store index of the instance in current frame predcitions, index in
        previous predictions, ID in previous predictions, IoU of the bboxes in this
        pair, period in previous predictions.

        Args:
            instances: D2 Instances, for predictions of the current frame
            iou_all: IoU for all bboxes pairs
        Return:
            A list of IoU for all pairs
        """
    bbox_pairs = []
    for i in range(len(instances)):
        for j in range(len(self._prev_instances)):
            bbox_pairs.append({'idx': i, 'prev_idx': j, 'prev_id': self.
                _prev_instances.ID[j], 'IoU': iou_all[i, j], 'prev_period':
                self._prev_instances.ID_period[j]})
    return bbox_pairs
