def activate(self, kalman_filter, frame_id):
    """Start a new tracklet."""
    self.kalman_filter = kalman_filter
    self.track_id = self.next_id()
    self.mean, self.covariance = self.kalman_filter.initiate(self.
        convert_coords(self._tlwh))
    self.tracklet_len = 0
    self.state = TrackState.Tracked
    if frame_id == 1:
        self.is_activated = True
    self.frame_id = frame_id
    self.start_frame = frame_id
