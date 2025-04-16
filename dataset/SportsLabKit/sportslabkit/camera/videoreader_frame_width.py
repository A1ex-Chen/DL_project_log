@property
def frame_width(self):
    return int(self._vc.get(cv2.CAP_PROP_FRAME_WIDTH))
