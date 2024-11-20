def get_dists(self, tracks, detections):
    """Get distances between tracks and detections using IoU and (optionally) ReID embeddings."""
    dists = matching.iou_distance(tracks, detections)
    dists_mask = dists > self.proximity_thresh
    dists = matching.fuse_score(dists, detections)
    if self.args.with_reid and self.encoder is not None:
        emb_dists = matching.embedding_distance(tracks, detections) / 2.0
        emb_dists[emb_dists > self.appearance_thresh] = 1.0
        emb_dists[dists_mask] = 1.0
        dists = np.minimum(dists, emb_dists)
    return dists
