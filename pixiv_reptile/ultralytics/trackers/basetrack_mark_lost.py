def mark_lost(self):
    """Mark the track as lost."""
    self.state = TrackState.Lost
