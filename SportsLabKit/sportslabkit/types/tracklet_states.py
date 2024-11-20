@property
def states(self) ->dict[str, Any]:
    """Get the current state of the tracker.

        Returns:
            Dict[str, Any]: A dictionary containing the current state of the tracker.
        """
    return self._states
