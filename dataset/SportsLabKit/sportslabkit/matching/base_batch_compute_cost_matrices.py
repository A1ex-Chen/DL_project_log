@abstractmethod
def compute_cost_matrices(self, trackers: list[Tracklet],
    list_of_detections: list[list[Detection]]) ->list[np.ndarray]:
    """Calculate the cost matrix between trackers and detections.

        Args:
            trackers: A list of trackers.
            list_of_detections: A list containing a list of detections for each frame.

        Returns:
            A list of 2D numpy arrays where the element at [i, j] in the kth array is the cost between tracker i and detection j in frame k.
        """
    pass
