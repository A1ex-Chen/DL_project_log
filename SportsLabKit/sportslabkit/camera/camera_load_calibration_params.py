def load_calibration_params(self):
    calibration_video_path = self.calibration_video_path
    if self.camera_matrix_path:
        np.load(self.camera_matrix_path)
    if self.distortion_coefficients_path:
        np.load(self.distortion_coefficients_path)
    if self.camera_matrix is None or self.distortion_coefficients is None:
        if calibration_video_path is not None:
            (self.camera_matrix, self.distortion_coefficients, self.mapx,
                self.mapy) = find_intrinsic_camera_parameters(
                calibration_video_path)
            self.camera_matrix_path = (calibration_video_path +
                '.camera_matrix.npy')
            self.distortion_coefficients_path = (calibration_video_path +
                '.distortion_coefficients.npy')
        else:
            self.camera_matrix = np.eye(3)
            self.distortion_coefficients = np.zeros(4)
            dim = self.frame_width, self.frame_height
            newcameramtx, _ = cv.getOptimalNewCameraMatrix(self.
                camera_matrix, self.distortion_coefficients, dim, 1, dim)
            self.mapx, self.mapy = cv.initUndistortRectifyMap(self.
                camera_matrix, self.distortion_coefficients, None,
                newcameramtx, dim, 5)
