def __init__(self, video_path: PathLike, threaded: bool=False, queue_size:
    int=10, keypoint_xml: (str | None)=None, x_range: (Sequence[float] |
    None)=(0, 105), y_range: (Sequence[float] | None)=(0, 68),
    camera_matrix: (ArrayLike | None)=None, camera_matrix_path: (str | None
    )=None, distortion_coefficients: (str | None)=None,
    distortion_coefficients_path: (str | None)=None, calibration_video_path:
    (str | None)=None, calibration_method: str='zhang', label: str='',
    verbose: int=0):
    """Class for handling camera calibration and undistortion.

        Args:
            video_path (str): path to video file.
            threaded (bool, optional): whether to use a threaded video reader. Defaults to False.
            queue_size (int, optional): size of queue for threaded video reader. Defaults to 10.
            keypoint_xml (str): path to file containing a mapping from pitch coordinates to video.
            x_range (Sequence[float]): pitch range to consider in x direction.
            y_range (Sequence[float]): pitch range to consider in y direction.
            camera_matrix (Optional[Union[str, np.ndarray]]): numpy array or path to file containing camera matrix.
            distortion_coefficients (Optional[Union[str, np.ndarray]]): numpy array or path to file containing distortion coefficients.
            calibration_video_path (Optional[str]): path to video file with checkerboard to use for calibration.
            label (str, optional): label for camera. Defaults to "".
            verbose (int, optional): verbosity level. Defaults to 0.
        Attributes:
            camera_matrix (np.ndarray): numpy array containing camera matrix.
            distortion_coefficients (np.ndarray): numpy array containing distortion coefficients.
            keypoint_map (Mapping): mapping from pitch coordinates to video.
            H (np.ndarray): homography matrix from image to pitch.
            w (int): width of video.
            h (int): height of video.

        """
    if threaded:
        logger.warning('Threaded video reader is buggy. Use at your own risk.')
    super().__init__(video_path, threaded, queue_size)
    self.label = label
    self.video_path = str(video_path)
    self.calibration_method = calibration_method
    self.camera_matrix = camera_matrix
    self.distortion_coefficients = distortion_coefficients
    self.camera_matrix_path = camera_matrix_path
    self.distortion_coefficients_path = distortion_coefficients_path
    self.calibration_video_path = calibration_video_path
    self.load_calibration_params()
    self.x_range = x_range
    self.y_range = y_range
    self.remove_leading_singleton = True
    if keypoint_xml is not None:
        source_keypoints, target_keypoints = read_pitch_keypoints(keypoint_xml,
            'video')
        self.source_keypoints = source_keypoints
        self.target_keypoints = target_keypoints
        source_keypoints = self.undistort_points(source_keypoints).squeeze()
        proj_error = np.linalg.norm(self.video2pitch(source_keypoints) -
            target_keypoints, axis=-1).mean()
        logger.debug(
            f'Camera `{self.label}`: projection error = {proj_error:.2f}m')
    else:
        self.source_keypoints = None
        self.target_keypoints = None
