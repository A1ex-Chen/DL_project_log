def update(self, sequence):
    list_of_detections = []
    for frame in sequence:
        detections = self.detection_model(frame)
        detections = detections[0].to_list()
        if len(detections) > 0 and self.image_model is not None:
            embeds = self.image_model.embed_detections(detections, frame)
            for i, det in enumerate(detections):
                det.feature = embeds[i]
        list_of_detections.append(detections)
    path = self.matching_fn([self.tracklet], list_of_detections)
    new_states = []
    for frame_idx, det_idx in enumerate(path):
        if det_idx >= 0:
            new_state = {'box': list_of_detections[frame_idx][det_idx].box,
                'score': list_of_detections[frame_idx][det_idx].score,
                'feature': list_of_detections[frame_idx][det_idx].feature}
        else:
            new_state = {'box': self.tracklet.box, 'score': 0.5, 'feature':
                self.tracklet.feature}
        if self.motion_model is not None:
            self.motion_model.update(new_state)
        new_states.append(new_state)
    return new_states
