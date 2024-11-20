def get_observation(self, name: (str | None)=None) ->(Any | None):
    """Get the most recent value of an observation type.

        Args:
            name (str): Name of the observation type.

        Returns:
            Optional[Any]: The most recent value of the specified observation type or None if not available.
        """
    if name is None:
        return [self._observations[name][-1] for name in self._observations]
    if name in self._observations and self._observations[name]:
        return self._observations[name][-1]
    return None
