def pre_initialize(self, initial_frame, detection_model, image_model,
    motion_model, matching_fn):
    self.detections = []
    self.detection_model = detection_model
    self.image_model = image_model
    self.matching_fn = matching_fn
    self.motion_model = motion_model
    if self.image_model is not None:
        self.target['feature'] = self.image_model.embed_detections([self.
            target], initial_frame)[0]
