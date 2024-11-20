@property
def H(self) ->NDArray[np.float64]:
    """Calculate the homography transformation matrix from pitch to video space.

        Returns:
            NDArray[np.float64]: homography transformation matrix.

        """
    H, *_ = cv.findHomography(self.source_keypoints, self.target_keypoints,
        cv.RANSAC, 5.0)
    return H
