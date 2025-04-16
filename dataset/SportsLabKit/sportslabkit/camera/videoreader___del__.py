def __del__(self):
    try:
        self._vr.release()
    except AttributeError:
        pass
