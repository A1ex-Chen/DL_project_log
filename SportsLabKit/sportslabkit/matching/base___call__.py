def __call__(self, trackers: Sequence[Tracklet], detections: Sequence[
    Detection], return_cost_matrix: bool=False) ->(np.ndarray | tuple[np.
    ndarray, np.ndarray]):
    """Calculate the matching cost between trackers and detections.

        Args:
            trackers: A list of trackers.
            detections: A list of detections.

        returns:
            An array of containing indices of matching pairs of trackers and detections.
        """
    _check_trackers(trackers)
    _check_detections(detections)
    cost_matrix = self.compute_cost_matrix(trackers, detections)
    _check_cost_matrix(cost_matrix, trackers, detections)
    matches = self.match_cost_matrix(cost_matrix)
    _check_matches(matches, trackers, detections)
    if return_cost_matrix:
        return matches, cost_matrix
    return matches
