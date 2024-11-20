def apply(self, raw_frame: np.array, detections: list=None) ->np.array:
    """
        Apply object detection on a raw frame using specified method.

        Args:
            raw_frame (np.ndarray): The raw frame to be processed.
            detections (list): List of detections to be used in the processing.

        Returns:
            (np.ndarray): Processed frame.

        Examples:
            >>> gmc = GMC()
            >>> gmc.apply(np.array([[1, 2, 3], [4, 5, 6]]))
            array([[1, 2, 3],
                   [4, 5, 6]])
        """
    if self.method in {'orb', 'sift'}:
        return self.applyFeatures(raw_frame, detections)
    elif self.method == 'ecc':
        return self.applyEcc(raw_frame)
    elif self.method == 'sparseOptFlow':
        return self.applySparseOptFlow(raw_frame)
    else:
        return np.eye(2, 3)
