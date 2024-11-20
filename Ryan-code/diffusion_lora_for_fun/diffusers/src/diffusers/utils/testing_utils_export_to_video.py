def export_to_video(video_frames: List[np.ndarray], output_video_path: str=None
    ) ->str:
    if is_opencv_available():
        import cv2
    else:
        raise ImportError(BACKENDS_MAPPING['opencv'][1].format(
            'export_to_video'))
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w, c = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=8,
        frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)
    return output_video_path
