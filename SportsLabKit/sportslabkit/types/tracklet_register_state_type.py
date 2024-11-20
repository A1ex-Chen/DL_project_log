def register_state_type(self, name: str) ->None:
    """Register a new state type.

        Args:
            name (str): Name of the new state type to be registered.
        """
    if name not in self._states:
        self._states[name] = None
