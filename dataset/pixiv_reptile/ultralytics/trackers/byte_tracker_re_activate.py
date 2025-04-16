def re_activate(self, new_track, frame_id, new_id=False):
    """Reactivates a previously lost track with a new detection."""
    self.mean, self.covariance = self.kalman_filter.update(self.mean, self.
        covariance, self.convert_coords(new_track.tlwh))
    self.tracklet_len = 0
    self.state = TrackState.Tracked
    self.is_activated = True
    self.frame_id = frame_id
    if new_id:
        self.track_id = self.next_id()
    self.score = new_track.score
    self.cls = new_track.cls
    self.angle = new_track.angle
    self.idx = new_track.idx
