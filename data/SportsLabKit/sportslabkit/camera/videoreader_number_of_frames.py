@property
def number_of_frames(self):
    return int(self._vc.get(cv2.CAP_PROP_FRAME_COUNT))
