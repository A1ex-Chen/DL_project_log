def feature_distance(detection_fst, unmatched_trackers):
    if detection_fst.embedding is None:
        return 10
    snd_embedding = unmatched_trackers.last_detection.embedding
    if snd_embedding is None:
        return 10
    return cosine_similarity(snd_embedding, detection_fst.embedding)
