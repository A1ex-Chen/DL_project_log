def get_dists(self, tracks, detections):
    """Calculates the distance between tracks and detections using IoU and fuses scores."""
    dists = matching.iou_distance(tracks, detections)
    dists = matching.fuse_score(dists, detections)
    return dists
