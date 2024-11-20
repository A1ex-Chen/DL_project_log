def applyFeatures(self, raw_frame: np.array, detections: list=None) ->np.array:
    """
        Apply feature-based methods like ORB or SIFT to a raw frame.

        Args:
            raw_frame (np.ndarray): The raw frame to be processed.
            detections (list): List of detections to be used in the processing.

        Returns:
            (np.ndarray): Processed frame.

        Examples:
            >>> gmc = GMC()
            >>> gmc.applyFeatures(np.array([[1, 2, 3], [4, 5, 6]]))
            array([[1, 2, 3],
                   [4, 5, 6]])
        """
    height, width, _ = raw_frame.shape
    frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
    H = np.eye(2, 3)
    if self.downscale > 1.0:
        frame = cv2.resize(frame, (width // self.downscale, height // self.
            downscale))
        width = width // self.downscale
        height = height // self.downscale
    mask = np.zeros_like(frame)
    mask[int(0.02 * height):int(0.98 * height), int(0.02 * width):int(0.98 *
        width)] = 255
    if detections is not None:
        for det in detections:
            tlbr = (det[:4] / self.downscale).astype(np.int_)
            mask[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]] = 0
    keypoints = self.detector.detect(frame, mask)
    keypoints, descriptors = self.extractor.compute(frame, keypoints)
    if not self.initializedFirstFrame:
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        self.prevDescriptors = copy.copy(descriptors)
        self.initializedFirstFrame = True
        return H
    knnMatches = self.matcher.knnMatch(self.prevDescriptors, descriptors, 2)
    matches = []
    spatialDistances = []
    maxSpatialDistance = 0.25 * np.array([width, height])
    if len(knnMatches) == 0:
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        self.prevDescriptors = copy.copy(descriptors)
        return H
    for m, n in knnMatches:
        if m.distance < 0.9 * n.distance:
            prevKeyPointLocation = self.prevKeyPoints[m.queryIdx].pt
            currKeyPointLocation = keypoints[m.trainIdx].pt
            spatialDistance = prevKeyPointLocation[0] - currKeyPointLocation[0
                ], prevKeyPointLocation[1] - currKeyPointLocation[1]
            if np.abs(spatialDistance[0]) < maxSpatialDistance[0] and np.abs(
                spatialDistance[1]) < maxSpatialDistance[1]:
                spatialDistances.append(spatialDistance)
                matches.append(m)
    meanSpatialDistances = np.mean(spatialDistances, 0)
    stdSpatialDistances = np.std(spatialDistances, 0)
    inliers = (spatialDistances - meanSpatialDistances < 2.5 *
        stdSpatialDistances)
    goodMatches = []
    prevPoints = []
    currPoints = []
    for i in range(len(matches)):
        if inliers[i, 0] and inliers[i, 1]:
            goodMatches.append(matches[i])
            prevPoints.append(self.prevKeyPoints[matches[i].queryIdx].pt)
            currPoints.append(keypoints[matches[i].trainIdx].pt)
    prevPoints = np.array(prevPoints)
    currPoints = np.array(currPoints)
    if prevPoints.shape[0] > 4:
        H, inliers = cv2.estimateAffinePartial2D(prevPoints, currPoints,
            cv2.RANSAC)
        if self.downscale > 1.0:
            H[0, 2] *= self.downscale
            H[1, 2] *= self.downscale
    else:
        LOGGER.warning('WARNING: not enough matching points')
    self.prevFrame = frame.copy()
    self.prevKeyPoints = copy.copy(keypoints)
    self.prevDescriptors = copy.copy(descriptors)
    return H
