def read(self, frame_number=None):
    """Read next frame or frame specified by `frame_number`."""
    if not self.stopped and self.threaded:
        sleep(10 ** -6)
        frame = self.q.get(0.1)
        return True, frame
    is_current_frame = frame_number == self.current_frame_pos
    if frame_number is not None and not is_current_frame:
        self._seek(frame_number)
    ret, frame = self._vc.read()
    return ret, frame
