def embedding_distance(matched_not_init_trackers, unmatched_trackers):
    snd_embedding = unmatched_trackers.last_detection.embedding
    if snd_embedding is None:
        for detection in reversed(unmatched_trackers.past_detections):
            if detection.embedding is not None:
                snd_embedding = detection.embedding
                break
        else:
            return 1
    for detection_fst in matched_not_init_trackers.past_detections:
        if detection_fst.embedding is None:
            continue
        distance = cosine_similarity(snd_embedding, detection_fst.embedding)
        if distance < config.reid_distance_threshold:
            return distance
    return 1
