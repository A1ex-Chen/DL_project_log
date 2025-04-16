def match_cost_matrix(self, cost_matrix: np.ndarray) ->np.ndarray:
    """Match trackers and detections based on a cost matrix.

        While this method implements a hungarian algorithm, it is can be
        overriden by subclasses that implement different matching strategies.
        Args:
            cost_matrix: A 2D numpy array of matching costs between trackers and detections.

        returns:
            A 2D numpy array of shape (n, 2) containing indices of matching pairs of trackers and detections.
        """
    matches = np.array(linear_sum_assignment_with_inf(cost_matrix)).T
    return matches
