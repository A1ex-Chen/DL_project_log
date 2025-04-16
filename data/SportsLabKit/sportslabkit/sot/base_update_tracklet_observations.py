def update_tracklet_observations(self, states: dict[str, Any]):
    self.check_required_types(states)
    for required_type in self.required_keys:
        self.tracklet.update_observation(required_type, states[required_type])
    self.tracklet.increment_counter()
