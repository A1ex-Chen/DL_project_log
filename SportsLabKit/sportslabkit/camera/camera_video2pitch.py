def video2pitch(self, pts: ArrayLike) ->NDArray[np.float64]:
    """Convert image coordinates to pitch coordinates.

        Args:
            video_pts (np.ndarray): points in image coordinate space

        Returns:
            np.ndarray: points in pitch coordinate

        """
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    pitch_pts = cv.perspectiveTransform(np.asarray([pts], dtype=np.float32),
        self.H)
    return pitch_pts
