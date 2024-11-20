def merge(self, tracked_object):
    """Merge with a not yet initialized TrackedObject instance"""
    self.reid_hit_counter = None
    self.hit_counter = self.initial_period * 2
    self.point_hit_counter = tracked_object.point_hit_counter
    self.last_distance = tracked_object.last_distance
    self.current_min_distance = tracked_object.current_min_distance
    self.last_detection = tracked_object.last_detection
    self.detected_at_least_once_points = (tracked_object.
        detected_at_least_once_points)
    self.filter = tracked_object.filter
    for past_detection in tracked_object.past_detections:
        self._conditionally_add_to_past_detections(past_detection)
