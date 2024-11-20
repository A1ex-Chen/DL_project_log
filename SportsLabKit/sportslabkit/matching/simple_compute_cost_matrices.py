def compute_cost_matrices(self, trackers: list[Tracklet],
    list_of_detections: list[list[Detection]]) ->list[np.ndarray]:
    """Calculate the cost matrix between trackers and detections.

        Args:
            trackers: A list of trackers.
            list_of_detections: A list containing a list of detections for each frame.

        Returns:
            A list of 2D numpy arrays where the element at [i, j] in the kth array is the cost between tracker i and detection j in frame 0, detection j-1 and detection j otherwise.
        """
    cost_matrices = []
    for k, detections in enumerate(list_of_detections):
        num_detections = len(detections)
        if k == 0:
            cost_matrix = np.zeros((len(trackers), num_detections))
            for i, tracker in enumerate(trackers):
                for j, detection in enumerate(detections):
                    cost_matrix[i, j] = np.linalg.norm(np.array(tracker.box
                        ) - np.array(detection.box))
        else:
            prev_detections = list_of_detections[k - 1]
            cost_matrix = np.zeros((len(prev_detections), num_detections))
            for j, detection in enumerate(detections):
                for i in range(min(len(prev_detections), j + 1)):
                    cost_matrix[i, j] = np.linalg.norm(np.array(
                        prev_detections[i].box) - np.array(detection.box))
        cost_matrices.append(cost_matrix)
    return cost_matrices
