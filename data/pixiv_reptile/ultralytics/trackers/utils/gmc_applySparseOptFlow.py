def applySparseOptFlow(self, raw_frame: np.array) ->np.array:
    """
        Apply Sparse Optical Flow method to a raw frame.

        Args:
            raw_frame (np.ndarray): The raw frame to be processed.

        Returns:
            (np.ndarray): Processed frame.

        Examples:
            >>> gmc = GMC()
            >>> gmc.applySparseOptFlow(np.array([[1, 2, 3], [4, 5, 6]]))
            array([[1, 2, 3],
                   [4, 5, 6]])
        """
    height, width, _ = raw_frame.shape
    frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
    H = np.eye(2, 3)
    if self.downscale > 1.0:
        frame = cv2.resize(frame, (width // self.downscale, height // self.
            downscale))
    keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params
        )
    if not self.initializedFirstFrame or self.prevKeyPoints is None:
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        self.initializedFirstFrame = True
        return H
    matchedKeypoints, status, _ = cv2.calcOpticalFlowPyrLK(self.prevFrame,
        frame, self.prevKeyPoints, None)
    prevPoints = []
    currPoints = []
    for i in range(len(status)):
        if status[i]:
            prevPoints.append(self.prevKeyPoints[i])
            currPoints.append(matchedKeypoints[i])
    prevPoints = np.array(prevPoints)
    currPoints = np.array(currPoints)
    if prevPoints.shape[0] > 4 and prevPoints.shape[0] == prevPoints.shape[0]:
        H, _ = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)
        if self.downscale > 1.0:
            H[0, 2] *= self.downscale
            H[1, 2] *= self.downscale
    else:
        LOGGER.warning('WARNING: not enough matching points')
    self.prevFrame = frame.copy()
    self.prevKeyPoints = copy.copy(keypoints)
    return H
