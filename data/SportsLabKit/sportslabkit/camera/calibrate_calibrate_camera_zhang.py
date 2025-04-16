def calibrate_camera_zhang(objpoints, imgpoints, dim):
    """Compute camera matrix and distortion coefficients using Zhang's method.

    Args:
        objpoints (list): A list of 3D points in real world space.
        imgpoints (list): A list of 2D points in the image plane.
        dim (tuple): The image dimensions.

    Returns:
        tuple: A tuple containing the camera matrix, distortion coefficients, rotation vectors, and translation vectors.
    """
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, dim,
        None, None)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, dim, 1, dim)
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newcameramtx, dim, 5)
    return K, D, mapx, mapy
