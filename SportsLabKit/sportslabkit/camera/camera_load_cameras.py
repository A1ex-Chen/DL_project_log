def load_cameras(camera_info: list[Mapping]) ->list[Camera]:
    """Load cameras from a list of dictionaries containing camera information.

    Args:
        camera_info (List[Mapping]): list of dictionaries containing camera information.

    Returns:
        List[Camera]: list of cameras objects.

    """
    cameras = []
    for cam_info in camera_info:
        camera = Camera(video_path=cam_info.video_path, keypoint_xml=
            cam_info.keypoint_xml, camera_matrix=cam_info.camera_matrix,
            camera_matrix_path=cam_info.camera_matrix_path,
            distortion_coefficients=cam_info.distortion_coefficients,
            distortion_coefficients_path=cam_info.
            distortion_coefficients_path, calibration_video_path=cam_info.
            calibration_video_path, x_range=cam_info.x_range, y_range=
            cam_info.y_range, label=cam_info.label)
        cameras.append(camera)
    return cameras
