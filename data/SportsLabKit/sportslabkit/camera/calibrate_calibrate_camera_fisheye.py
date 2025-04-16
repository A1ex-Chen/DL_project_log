def calibrate_camera_fisheye(objpoints, imgpoints, dim, balance=1):
    """Compute camera matrix and distortion coefficients using fisheye method.

    Args:
        objpoints (list): A list of 3D points in real world space.
        imgpoints (list): A list of 2D points in the image plane.
        dim (tuple): The image dimensions.
        balance (float): The balance factor. Must be between 0 and 1. Larger values wil

    Returns:
        tuple: A tuple containing the camera matrix, distortion coefficients, rotation vectors, and translation vectors.
    """
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    objpoints = np.expand_dims(np.asarray(objpoints), -2)
    ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints,
        dim, K, D, rvecs, tvecs, cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
        cv2.fisheye.CALIB_FIX_SKEW, (cv2.TERM_CRITERIA_EPS + cv2.
        TERM_CRITERIA_MAX_ITER, 30, 1e-06))
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D,
        dim, np.eye(3), balance=2)
    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K,
        dim, cv2.CV_32FC1)
    return K, D, mapx, mapy
