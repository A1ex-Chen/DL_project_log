def get_observations(self, name: (str | None)=None) ->(Any | None):
    """Get all values of an observation type.

        Args:
            name (str): Name of the observation type.

        Returns:
            List[Any]: All values of the specified observation type.
        """
    if name is None:
        return self._observations
    if name in self._observations:
        return self._observations[name]
    else:
        raise ValueError(f"Observation type '{name}' not registered")
