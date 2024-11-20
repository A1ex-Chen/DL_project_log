@property
def frame_channels(self):
    n_channels = int(self._vc.get(cv2.CAP_PROP_CHANNEL))
    if n_channels == 0:
        self._reset()
        n_channels = self.read(0)[1].shape[-1]
    return n_channels
