def update(self, current_frame, tracklets):
    detections = self.detection_model(current_frame)
    for i, tracklet in enumerate(tracklets):
        predicted_box = self.motion_model(tracklet)
        tracklet.update_state('pred_box', predicted_box)
    detections = detections[0].to_list()
    matches, cost_matrix = self.matching_fn(tracklets, detections,
        return_cost_matrix=True)
    assigned_tracklets = []
    new_tracklets = []
    unassigned_tracklets = []
    for match in matches:
        track_idx, det_idx = match[0], match[1]
        tracklet = tracklets[track_idx]
        logger.debug(
            f"track_idx: {track_idx}, det_idx: {det_idx}, cost: {cost_matrix[track_idx, det_idx]}, track staleness: {tracklet.get_state('staleness')}"
            )
        new_observation = {'box': detections[det_idx].box, 'score':
            detections[det_idx].score, 'frame': self.frame_count}
        tracklet = self.update_tracklet(tracklet, new_observation)
        assigned_tracklets.append(tracklet)
    for i, det in enumerate(detections):
        if i not in [match[1] for match in matches]:
            new_observation = {'box': det.box, 'score': det.score, 'frame':
                self.frame_count}
            new_tracklet = self.create_tracklet(new_observation)
            new_tracklets.append(new_tracklet)
    for i, tracklet in enumerate(tracklets):
        if i not in [match[0] for match in matches]:
            new_observation = {'box': tracklet.get_state('pred_box'),
                'score': tracklet.get_observation('score'), 'frame': self.
                frame_count}
            tracklet = self.update_tracklet(tracklet, new_observation)
            unassigned_tracklets.append(tracklet)
    return assigned_tracklets, new_tracklets, unassigned_tracklets
