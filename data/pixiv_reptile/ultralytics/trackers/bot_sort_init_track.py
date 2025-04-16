def init_track(self, dets, scores, cls, img=None):
    """Initialize track with detections, scores, and classes."""
    if len(dets) == 0:
        return []
    if self.args.with_reid and self.encoder is not None:
        features_keep = self.encoder.inference(img, dets)
        return [BOTrack(xyxy, s, c, f) for xyxy, s, c, f in zip(dets,
            scores, cls, features_keep)]
    else:
        return [BOTrack(xyxy, s, c) for xyxy, s, c in zip(dets, scores, cls)]
