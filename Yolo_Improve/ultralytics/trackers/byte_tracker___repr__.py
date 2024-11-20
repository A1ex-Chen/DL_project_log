def __repr__(self):
    """Return a string representation of the BYTETracker object with start and end frames and track ID."""
    return f'OT_{self.track_id}_({self.start_frame}-{self.end_frame})'
