def _check_required_observations(self, tracklet: Tracklet) ->None:
    """Check if the required observations are registered in the SingleObjectTracker instance.

        Args:
            sot (SingleObjectTracker): The single object tracker instance.

        Raises:
            KeyError: If a required observation is not registered in the SingleObjectTracker instance.
        """
    for obs_type in self.required_observation_types:
        if obs_type not in tracklet._observations:
            raise KeyError(
                f'{self.name} requires observation type `{obs_type}` but it is not registered.'
                )
        if len(tracklet._observations[obs_type]) == 0:
            raise KeyError(
                f'{self.name} requires observation type `{obs_type}` but it is empty.'
                )
