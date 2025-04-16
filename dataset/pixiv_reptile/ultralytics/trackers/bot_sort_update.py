def update(self, new_track, frame_id):
    """Update the YOLOv8 instance with new track and frame ID."""
    if new_track.curr_feat is not None:
        self.update_features(new_track.curr_feat)
    super().update(new_track, frame_id)
