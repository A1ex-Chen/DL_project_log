def get_state(self, name: (str | None)=None) ->(Any | None):
    """Get the most recent value of a state type.

        Args:
            name (str): Name of the state type.

        Returns:
            Optional[Any]: The most recent value of the specified state type or None if not available.
        """
    if name is None:
        return [self._states[name] for name in self._states]
    if name in self._states:
        return self._states[name]
    return None
