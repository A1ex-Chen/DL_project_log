@property
def frame_rate(self):
    return self._vc.get(cv2.CAP_PROP_FPS)
