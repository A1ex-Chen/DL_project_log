@abstractmethod
def compute_cost_matrix(self, trackers: Sequence[Tracklet], detections:
    Sequence[Detection]) ->np.ndarray:
    """Calculate the matching cost between trackers and detections.

        Args:
            trackers: A list of trackers.
            detections: A list of detections.

        returns:
            A 2D numpy array of matching costs between trackers and detections.
        """
    pass
