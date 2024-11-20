@property
def fourcc(self):
    return int(self._vc.get(cv2.CAP_PROP_FOURCC))
