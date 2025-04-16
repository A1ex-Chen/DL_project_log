def compute_cost_matrix(self, trackers: Sequence[Tracklet], detections:
    Sequence[Detection]) ->np.ndarray:
    """Calculate the matching cost between trackers and detections.

        Args:
            trackers: A list of trackers.
            detections: A list of detections.

        returns:
            A 2D numpy array of matching costs between trackers and detections.
        """
    if len(trackers) == 0 or len(detections) == 0:
        return np.array([])
    cost_matrix = self.metric(trackers, detections)
    cost_matrix = cost_matrix
    cost_matrix[cost_matrix > self.gate] = np.inf
    return cost_matrix
