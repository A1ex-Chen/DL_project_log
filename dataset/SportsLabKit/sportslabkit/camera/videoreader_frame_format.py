@property
def frame_format(self):
    return int(self._vc.get(cv2.CAP_PROP_FORMAT))
