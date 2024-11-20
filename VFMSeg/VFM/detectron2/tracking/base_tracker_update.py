def update(self, predictions: Instances) ->Instances:
    """
        Args:
            predictions: D2 Instances for predictions of the current frame
        Return:
            D2 Instances for predictions of the current frame with ID assigned

        _prev_instances and instances will have the following fields:
          .pred_boxes               (shape=[N, 4])
          .scores                   (shape=[N,])
          .pred_classes             (shape=[N,])
          .pred_keypoints           (shape=[N, M, 3], Optional)
          .pred_masks               (shape=List[2D_MASK], Optional)   2D_MASK: shape=[H, W]
          .ID                       (shape=[N,])

        N: # of detected bboxes
        H and W: height and width of 2D mask
        """
    raise NotImplementedError('Calling BaseTracker::update')
