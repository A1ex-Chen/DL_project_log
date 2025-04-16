def __call__(self, trackers: Sequence[Tracklet], detections: Sequence[
    Detection]) ->np.ndarray:
    """Calculate the metric between trackers and detections.

        Args:
            trackers: A list of trackers.
            detections: A list of detections.

        returns:
            An array of containing the metric between trackers and detections.
        """
    _check_trackers(trackers)
    _check_detections(detections)
    cost_matrix = self.compute_metric(trackers, detections)
    _check_cost_matrix(cost_matrix, trackers, detections)
    return cost_matrix
