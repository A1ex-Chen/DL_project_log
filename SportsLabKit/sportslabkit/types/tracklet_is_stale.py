def is_stale(self) ->bool:
    """Check if the tracker is stale.

        Returns:
            bool: True if the tracker's staleness is greater than max_staleness, otherwise False.
        """
    return self.staleness > self.max_staleness
