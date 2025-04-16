def __init__(self):
    """Initializes a new track with unique ID and foundational tracking attributes."""
    self.track_id = 0
    self.is_activated = False
    self.state = TrackState.New
    self.history = OrderedDict()
    self.features = []
    self.curr_feature = None
    self.score = 0
    self.start_frame = 0
    self.frame_id = 0
    self.time_since_update = 0
    self.location = np.inf, np.inf
