def update_tracklet(self, tracklet: Tracklet, states: dict[str, Any]):
    self._check_required_observations(states)
    tracklet.update_observations(states, self.frame_count)
    tracklet.increment_counter()
    return tracklet
