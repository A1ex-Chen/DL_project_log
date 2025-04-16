def export_to_video(video_frames: Union[List[np.ndarray], List[PIL.Image.
    Image]], output_video_path: str=None, fps: int=10) ->str:
    if is_opencv_available():
        import cv2
    else:
        raise ImportError(BACKENDS_MAPPING['opencv'][1].format(
            'export_to_video'))
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix='.mp4').name
    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in
            video_frames]
    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w, c = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps,
        frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)
    return output_video_path
