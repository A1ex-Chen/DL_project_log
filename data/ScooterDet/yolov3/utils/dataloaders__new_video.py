def _new_video(self, path):
    self.frame = 0
    self.cap = cv2.VideoCapture(path)
    self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
    self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))
