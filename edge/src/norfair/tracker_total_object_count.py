@property
def total_object_count(self) ->int:
    """Total number of TrackedObjects initialized in the by this Tracker"""
    return self._obj_factory.count
