def _seek(self, frame_number):
    """Go to frame."""
    self._vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
