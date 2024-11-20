def update(self, current_frame):
    current_frame = current_frame[0]
    detections = self.detection_model(current_frame)
    if self.motion_model is not None:
        predictions = self.motion_model(self.tracklet)
        self.tracklet.update_current_observation('box', predictions)
    detections = detections[0].to_list()
    if len(detections) > 0 and self.image_model is not None:
        embeds = self.image_model.embed_detections(detections, current_frame)
        for i, det in enumerate(detections):
            det.feature = embeds[i]
    match = self.matching_fn([self.tracklet], detections)
    if len(match) > 0:
        _, det_idx = match[0][0], match[0][1]
        new_state = {'box': detections[det_idx].box, 'score': detections[
            det_idx].score, 'feature': detections[det_idx].feature}
    else:
        new_state = {'box': self.tracklet.box, 'score': 0.5, 'feature':
            self.tracklet.feature}
    if self.motion_model is not None:
        self.motion_model.update(new_state)
    return new_state
