def _new_video(self, path):
    """Creates a new video capture object for the given path."""
    self.frame = 0
    self.cap = cv2.VideoCapture(path)
    self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
    if not self.cap.isOpened():
        raise FileNotFoundError(f'Failed to open video {path}')
    self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
