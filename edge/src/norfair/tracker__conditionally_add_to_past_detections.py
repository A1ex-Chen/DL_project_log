def _conditionally_add_to_past_detections(self, detection):
    """Adds detections into (and pops detections away) from `past_detections`

        It does so by keeping a fixed amount of past detections saved into each
        TrackedObject, while maintaining them distributed uniformly through the object's
        lifetime.
        """
    if self.past_detections_length == 0:
        return
    if len(self.past_detections) < self.past_detections_length:
        detection.age = self.age
        self.past_detections.append(detection)
    elif self.age >= self.past_detections[0].age * self.past_detections_length:
        self.past_detections.pop(0)
        detection.age = self.age
        self.past_detections.append(detection)
