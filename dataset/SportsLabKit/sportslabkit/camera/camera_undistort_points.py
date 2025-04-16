def undistort_points(self, points: ArrayLike) ->NDArray[np.float64]:
    """Undistort points with the camera matrix and distortion coefficients.

        Args:
            points (ArrayLike): [description]

        Returns:
            NDArray[np.float64]: [description]

        Note:
            Not to be confused with video2pitch which uses a homography transformation.
        """
    mtx = self.camera_matrix
    dist = self.distortion_coefficients
    w = self.w
    h = self.h
    if self.calibration_method == 'zhang':
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h),
            1, (w, h))
        dst = cv.undistortPoints(points, mtx, dist, None, newcameramtx)
        dst = dst.reshape(-1, 2)
        x, y, w, h = roi
        dst = dst - np.asarray([x, y])
    elif self.calibration_method == 'fisheye':
        mtx_new = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(mtx,
            dist, (w, h), np.eye(3), balance=1.0)
        points = np.expand_dims(points, axis=1)
        dst = np.squeeze(cv.fisheye.undistortPoints(points, mtx, dist, P=
            mtx_new))
    return dst
