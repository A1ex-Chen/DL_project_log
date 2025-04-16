def find_intrinsic_camera_parameters(media_path: PathLike, fps: int=1,
    scale: int=4, save_path: (PathLike | None)=None, draw_on_save: bool=
    False, points_to_use: int=50, calibration_method: str='zhang',
    return_mappings: bool=True) ->tuple[np.ndarray, np.ndarray, np.ndarray,
    np.ndarray]:
    """Calculate the intrinsic parameters of a camera from a video of a checkerboard pattern.

    This function takes a video file containing a checkerboard pattern and calculates the intrinsic parameters of the camera. The video is first processed to locate the corners of the checkerboard in each frame. These corners are then used to compute the intrinsic parameters of the camera.

    Args:
        media_path (Union[str, Path]): Path to the video file or a list of video files containing the checkerboard pattern. Wildcards are supported.
        fps (int, optional): Frames per second to use when processing the video. Defaults to 1.
        scale (int, optional): Scale factor to use when processing the video. Defaults to 4.
        save_path (Optional[Union[str, Path]], optional): Path to save the computed intrinsic parameters. If not specified, the parameters are not saved. Defaults to None.
        draw_on_save (bool, optional): If `True`, the corners of the checkerboard are drawn on the frames and saved with the intrinsic parameters. Defaults to False.
        points_to_use (int, optional): Number of frames to use when calculating the intrinsic parameters. If more frames are found than this number, a subset of frames is selected based on their location in the image plane. Defaults to 50.
        calibration_method (str, optional): Calibration method to use. Must be either "zhang" or "fisheye". Defaults to "zhang".
        return_mappings (bool, optional): If `True`, the function returns the computed mapping functions along with the intrinsic parameters. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the camera matrix, distortion coefficients, and mapping functions (if `return_mappings` is True).

    Raises:
        ValueError: If the `calibration_method` is not "zhang" or "fisheye".
    """
    from sportslabkit.camera import Camera
    camera = Camera(media_path)
    objpoints, imgpoints = detect_corners(camera, scale, fps)
    if len(imgpoints) == 0:
        logger.error('No checkerboards found.')
    logger.info(f'imgpoints found: {len(imgpoints)}')
    imgpoints, objpoints = select_images(imgpoints, objpoints, points_to_use)
    logger.debug(f'imgpoints used: {len(imgpoints)}')
    if 1 <= points_to_use <= len(imgpoints):
        logger.info(
            f'Too many ({len(imgpoints)}) checkerboards found. Selecting {points_to_use}.'
            )
    logger.info('Computing calibration parameters...')
    dim = camera.frame_width, camera.frame_height
    if calibration_method.lower() == 'zhang':
        logger.info("Using Zhang's method.")
        K, D, mapx, mapy = calibrate_camera_zhang(objpoints, imgpoints, dim)
    elif calibration_method.lower() == 'fisheye':
        K, D, mapx, mapy = calibrate_camera_fisheye(objpoints, imgpoints, dim)
    else:
        raise ValueError('Calibration method must be `zhang` or `fisheye`.')
    logger.info('Finished computing calibration parameters.')
    return K, D, mapx, mapy
