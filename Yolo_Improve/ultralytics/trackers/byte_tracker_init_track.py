def init_track(self, dets, scores, cls, img=None):
    """Initialize object tracking with detections and scores using STrack algorithm."""
    return [STrack(xyxy, s, c) for xyxy, s, c in zip(dets, scores, cls)
        ] if len(dets) else []
