def update_current_observation(self, name: str, value: Any) ->None:
    """Update the most recent observation with a new value.

        Args:
            name (str): Name of the observation type to be updated.
            value (Any): New value for the specified observation type.
        """
    if name in self._observations:
        self._observations[name][-1] = value
    else:
        raise ValueError(f"Observation type '{name}' not registered")
