def is_active(self) ->bool:
    """Check if the tracker is active.

        Returns:
            bool: True if the tracker is active (i.e. steps_alive > 0, not stale, and not invalid), otherwise False.
        """
    return self.steps_alive > 0 and not self.is_stale()
