def calibrate_video_from_mappings(media_path: PathLike, mapx: NDArray, mapy:
    NDArray, save_path: PathLike, stabilize: bool=True):
    """
    Calibrates a video using provided mapping parameters.

    Args:
    media_path (str): The path to the input video file.
    mapx (NDArray): The mapping array for x-axis.
    mapy (NDArray): The mapping array for y-axis.
    save_path (str): The path to save the calibrated video.
    stabilize (bool, optional): Whether to stabilize the video or not. Default is True.

    Returns:
    None
    """

    def generator():
        stab = Stabilizer()
        from sportslabkit.camera import Camera
        camera = Camera(media_path)
        for frame in camera:
            stab_frame = stab.stabilize(frame)
            if stab_frame is not None and stabilize:
                frame = stab_frame
            frame = cv2.remap(frame, mapx, mapy, interpolation=cv2.
                INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            yield frame
    make_video(generator(), save_path)
