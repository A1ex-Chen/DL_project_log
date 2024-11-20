def create_tracklet(self, state: dict[str, Any]):
    tracklet = Tracklet(max_staleness=self.max_staleness)
    for required_type in self.required_observation_types:
        tracklet.register_observation_type(required_type)
    for required_type in self.required_state_types:
        tracklet.register_state_type(required_type)
    self._check_required_observations(state)
    self.update_tracklet(tracklet, state)
    return tracklet
