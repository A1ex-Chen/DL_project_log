def update(self, current_frame, tracklets):
    detections = self.detection_model(current_frame)
    for i, tracklet in enumerate(tracklets):
        predicted_box = self.motion_model(tracklet)
        tracklet.update_state('pred_box', predicted_box)
    detections = detections[0].to_list()
    if len(detections) > 0:
        embeds = self.image_model.embed_detections(detections, current_frame)
        for i, det in enumerate(detections):
            det.feature = embeds[i]
    high_confidence_detections = []
    low_confidence_detections = []
    for detection in detections:
        if detection.score > self.detection_score_threshold:
            high_confidence_detections.append(detection)
        else:
            low_confidence_detections.append(detection)
    logger.debug(
        f'd_high: {len(high_confidence_detections)}, d_low: {len(low_confidence_detections)}'
        )
    new_tracklets = []
    assigned_tracklets = []
    unassigned_tracklets = []
    matches_first, cost_matrix_first = self.first_matching_fn(tracklets,
        high_confidence_detections, True)
    for match in matches_first:
        track_idx, det_idx = match[0], match[1]
        logger.debug(
            f'track_idx: {track_idx}, det_idx: {det_idx}, cost: {cost_matrix_first[track_idx, det_idx]}'
            )
        det = high_confidence_detections[det_idx]
        tracklet = tracklets[track_idx]
        new_observation = {'box': det.box, 'score': det.score, 'feature':
            det.feature, 'frame': self.frame_count}
        tracklet = self.update_tracklet(tracklet, new_observation)
        assigned_tracklets.append(tracklet)
    for i, det in enumerate(high_confidence_detections):
        if i not in [match[1] for match in matches_first]:
            new_observation = {'box': det.box, 'score': det.score, 'frame':
                self.frame_count, 'feature': det.feature}
            new_tracklet = self.create_tracklet(new_observation)
            new_tracklets.append(new_tracklet)
    for i, tracklet in enumerate(tracklets):
        if i not in [match[0] for match in matches_first]:
            unassigned_tracklets.append(tracklet)
    matches_second, cost_matrix_second = self.second_matching_fn(
        unassigned_tracklets, low_confidence_detections, True)
    for match in matches_second:
        track_idx, det_idx = match[0], match[1]
        logger.debug(
            f'track_idx: {track_idx}, det_idx: {det_idx}, cost: {cost_matrix_second[track_idx, det_idx]}'
            )
        det = low_confidence_detections[det_idx]
        tracklet = unassigned_tracklets[track_idx]
        new_observation = {'box': det.box, 'score': det.score, 'feature':
            det.feature, 'frame': self.frame_count}
        tracklet = self.update_tracklet(tracklet, new_observation)
        assigned_tracklets.append(tracklet)
    unassigned_tracklets_second = []
    for i, tracklet in enumerate(unassigned_tracklets):
        if i not in [match[0] for match in matches_second]:
            new_observation = {'box': tracklet.get_state('pred_box'),
                'score': tracklet.get_observation('score'), 'frame': self.
                frame_count, 'feature': tracklet.get_observation('feature')}
            tracklet = self.update_tracklet(tracklet, new_observation)
            unassigned_tracklets_second.append(tracklet)
    logger.debug(
        f'1st matches: {len(matches_first)}, 2nd matches: {len(matches_second)}'
        )
    return assigned_tracklets, new_tracklets, unassigned_tracklets_second
