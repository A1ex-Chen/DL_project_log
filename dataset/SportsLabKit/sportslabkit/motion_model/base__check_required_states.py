def _check_required_states(self, tracklet: Tracklet) ->None:
    """Check if the required states are registered in the SingleObjectTracker instance.

        Args:
            sot (SingleObjectTracker): The single object tracker instance.

        Raises:
            KeyError: If a required state is not registered in the SingleObjectTracker instance.
        """
    for state in self.required_state_types:
        if state not in tracklet._states:
            tracklet.register_state_type(state)
