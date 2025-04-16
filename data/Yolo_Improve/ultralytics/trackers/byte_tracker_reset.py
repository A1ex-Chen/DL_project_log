def reset(self):
    """Reset tracker."""
    self.tracked_stracks = []
    self.lost_stracks = []
    self.removed_stracks = []
    self.frame_id = 0
    self.kalman_filter = self.get_kalmanfilter()
    self.reset_id()
