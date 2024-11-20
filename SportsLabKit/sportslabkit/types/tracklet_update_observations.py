def update_observations(self, observations: dict[str, Any], global_step: (
    int | None)=None) ->None:
    for name, value in observations.items():
        self.update_observation(name, value)
