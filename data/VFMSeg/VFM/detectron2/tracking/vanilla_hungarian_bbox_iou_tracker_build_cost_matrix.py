def build_cost_matrix(self, instances: Instances, prev_instances: Instances
    ) ->np.ndarray:
    """
        Build the cost matrix for assignment problem
        (https://en.wikipedia.org/wiki/Assignment_problem)

        Args:
            instances: D2 Instances, for current frame predictions
            prev_instances: D2 Instances, for previous frame predictions

        Return:
            the cost matrix in numpy array
        """
    assert instances is not None and prev_instances is not None
    iou_all = pairwise_iou(boxes1=instances.pred_boxes, boxes2=self.
        _prev_instances.pred_boxes)
    bbox_pairs = create_prediction_pairs(instances, self._prev_instances,
        iou_all, threshold=self._track_iou_threshold)
    cost_matrix = np.full((len(instances), len(prev_instances)),
        LARGE_COST_VALUE)
    return self.assign_cost_matrix_values(cost_matrix, bbox_pairs)
