def re_activate(self, new_track, frame_id, new_id=False):
    """Reactivates a track with updated features and optionally assigns a new ID."""
    if new_track.curr_feat is not None:
        self.update_features(new_track.curr_feat)
    super().re_activate(new_track, frame_id, new_id)
