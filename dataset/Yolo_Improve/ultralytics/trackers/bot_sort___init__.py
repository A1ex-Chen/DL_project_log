def __init__(self, args, frame_rate=30):
    """Initialize YOLOv8 object with ReID module and GMC algorithm."""
    super().__init__(args, frame_rate)
    self.proximity_thresh = args.proximity_thresh
    self.appearance_thresh = args.appearance_thresh
    if args.with_reid:
        self.encoder = None
    self.gmc = GMC(method=args.gmc_method)
