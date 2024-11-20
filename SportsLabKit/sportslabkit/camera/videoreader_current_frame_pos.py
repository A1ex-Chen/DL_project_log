@property
def current_frame_pos(self):
    return int(self._vc.get(cv2.CAP_PROP_POS_FRAMES))
