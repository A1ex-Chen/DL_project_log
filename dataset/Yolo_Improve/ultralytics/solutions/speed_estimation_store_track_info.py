def store_track_info(self, track_id, box):
    """
        Stores track data.

        Args:
            track_id (int): Object track id.
            box (list): Object bounding box data.

        Returns:
            (list): Updated tracking history for the given track_id.
        """
    track = self.trk_history[track_id]
    bbox_center = float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)
    track.append(bbox_center)
    if len(track) > 30:
        track.pop(0)
    self.trk_pts = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
    return track
