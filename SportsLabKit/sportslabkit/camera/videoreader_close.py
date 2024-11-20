def close(self):
    """Close video file."""
    self._vc.release()
