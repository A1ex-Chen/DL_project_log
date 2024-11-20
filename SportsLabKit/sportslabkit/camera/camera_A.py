@property
def A(self) ->NDArray[np.float64]:
    """Calculate the affine transformation matrix from pitch to video space.

        Returns:
            NDArray[np.float64]: affine transformation matrix.

        """
    A, *_ = cv.estimateAffinePartial2D(self.source_keypoints, self.
        target_keypoints)
    return A
