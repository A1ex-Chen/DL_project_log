def detect_corners(camera, scale: float, fps: float, num_corners_x: int=5,
    num_corners_y: int=9):
    """Detects the corners in a set of images.

    This function detects the corners in a set of images using the cv2.findChessboardCorners function. The input images are
    downsampled using the scale parameter to increase processing speed. The function returns a tuple of two lists,
    `objpoints` and `imgpoints`, containing 3D points in real world space and 2D points in the image plane, respectively.

    Args:
        camera (Camera): A `Camera` object representing the camera from which the images were captured.
        scale (float): The scale to resize the images for faster detection. Must be greater than 0.
        fps (float): The frames per second to process. Must be greater than 0.
        num_corners_x (int, optional): The number of corners along the x-axis in the checkerboard pattern. Defaults to 5.
        num_corners_y (int, optional): The number of corners along the y-axis in the checkerboard pattern. Defaults to 9.

    Returns:
        tuple: A tuple containing two lists, `objpoints` and `imgpoints`. `objpoints` is a list of 3D points in real world space, and `imgpoints` is a list of 2D points in the image plane.

    Raises:
        ValueError: If `scale` or `fps` is less than or equal to 0.
        AssertionError: If no images are found in the video.
    """
    if scale <= 0:
        raise ValueError('The scale must be greater than 0.')
    if fps <= 0:
        raise ValueError('The fps must be greater than 0.')
    n_frames = len(camera)
    assert n_frames > 0, 'No images found in video.'
    nskip = np.ceil(camera.frame_rate / fps)
    objpoints = []
    imgpoints = []
    objp = np.zeros((num_corners_x * num_corners_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_corners_y, 0:num_corners_x].T.reshape(-1, 2)
    criteria = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001
    for i in tqdm(range(n_frames)):
        if i % nskip != 0:
            continue
        frame = camera[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, None, fx=1 / scale, fy=1 / scale)
        ret, corners = cv2.findChessboardCorners(gray_small, (num_corners_y,
            num_corners_x))
        if ret:
            corners *= scale
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                criteria)
            imgpoints.append(corners)
            objpoints.append(objp)
    return objpoints, imgpoints
