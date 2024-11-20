@property
def frame_height(self):
    return int(self._vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
