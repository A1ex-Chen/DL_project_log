def get_active_objects(self) ->List['TrackedObject']:
    """Get the list of active objects

        Returns
        -------
        List["TrackedObject"]
            The list of active objects
        """
    return [o for o in self.tracked_objects if not o.is_initializing and o.
        hit_counter_is_positive]
