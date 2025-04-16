def update_state(self, name: str, value: Any) ->None:
    """Update the state with a new value.

        Args:
            name (str): Name of the state to be updated.
            value (Any): New value for the specified state.
        """
    if name in self._states:
        self._states[name] = value
    else:
        raise ValueError(f"State type '{name}' not registered")
