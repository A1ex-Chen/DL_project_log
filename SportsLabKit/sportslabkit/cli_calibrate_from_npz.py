def calibrate_from_npz(self, input: str, npzfile: str, output: str,
    calibration_method: str='zhang', keypoint_xml: (str | None)=None, **kwargs
    ):
    """Calibrate a video using precomputed calibration parameters

        Args:
            input (str): _description_
            npzfile (str): _description_
            output (str): _description_
            calibration_method (str, optional): _description_. Defaults to "zhang".
            keypoint_xml (Optional[str], optional): _description_. Defaults to None.

        Note:
            kwargs are passed to `make_video`, so it is recommended that you refere to the documentation for `make_video`.

        """
    mtx, dist, mapx, mapy = np.load(npzfile).values()
    camera = Camera(video_path=input, keypoint_xml=keypoint_xml, x_range=
        None, y_range=None, calibration_method=calibration_method,
        camera_matrix=mtx, distortion_coefficients=dist)
    camera.mapx = mapx
    camera.mapy = mapy
    if keypoint_xml is not None:
        camera.source_keypoints = camera.undistort_points(camera.
            source_keypoints)
    dirname = os.path.dirname(output)
    if len(dirname) != 0:
        os.makedirs(dirname, exist_ok=True)
    camera.save_calibrated_video(save_path=output, **kwargs)
    logger.info(f'Video saved to {output}')
