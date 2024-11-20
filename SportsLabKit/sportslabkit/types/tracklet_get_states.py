def get_states(self, name: (str | None)=None) ->(Any | None):
    """Get all values of a state type.

        Args:
            name (str): Name of the state type.

        Returns:
            List[Any]: All values of the specified state type.
        """
    if name is None:
        return self._states
    if name in self._states:
        return self._states[name]
    else:
        raise ValueError(f"State type '{name}' not registered")
