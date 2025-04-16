def calibrate(self, input: str, checkerboard: str, output: str, fps: int=1,
    scale: int=1, pts: int=50, calibration_method: str='zhang',
    keypoint_xml: (str | None)=None):
    """Calibrate video from input

        Args:
            input (str): Path to the input video (wildcards are supported).
            checkerboard (str): Path to the checkerboard video (wildcards are supported).
            output (str): Path to the output video.
            fps (int, optional): Number of frames per second to use for calibration. Defaults to 1.
            scale (int, optional): Scale factor for the checkerboard. Scales the checkerboard video by 1/s. Defaults to 1.
            pts (int, optional): Number of points to use for calibration. Defaults to 50.
            calibration_method (str, optional): Calibration method. Defaults to "zhang".
            keypoint_xml (Optional[str], optional): Path to the keypoint xml file. Defaults to None.
        """
    input_files = list(glob(input))
    checkerboard_files = list(glob(checkerboard))
    mtx, dist, mapx, mapy = find_intrinsic_camera_parameters(checkerboard_files
        , fps=fps, scale=scale, save_path=False, draw_on_save=False,
        points_to_use=pts, calibration_method=calibration_method,
        return_mappings=True)
    for input_file in input_files:
        camera = Camera(video_path=input_file, keypoint_xml=keypoint_xml,
            x_range=None, y_range=None, calibration_method=
            calibration_method, calibration_video_path=checkerboard_files,
            camera_matrix=mtx, distortion_coefficients=dist)
        camera.mapx = mapx
        camera.mapy = mapy
        camera.source_keypoints = camera.undistort_points(camera.
            source_keypoints)
        save_path = os.path.join(output, os.path.basename(input_file))
        camera.save_calibrated_video(save_path=save_path)
        logger.info(f'Video saved to {save_path}')
