@property
def current_object_count(self) ->int:
    """Number of active TrackedObjects"""
    return len(self.get_active_objects())
