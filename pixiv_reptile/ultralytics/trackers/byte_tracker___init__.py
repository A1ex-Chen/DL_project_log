def __init__(self, args, frame_rate=30):
    """Initialize a YOLOv8 object to track objects with given arguments and frame rate."""
    self.tracked_stracks = []
    self.lost_stracks = []
    self.removed_stracks = []
    self.frame_id = 0
    self.args = args
    self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
    self.kalman_filter = self.get_kalmanfilter()
    self.reset_id()
