def register_observation_type(self, name: str) ->None:
    """Register a new observation type.

        Args:
            name (str): Name of the new observation type to be registered.
        """
    if name not in self._observations:
        self._observations[name] = []
