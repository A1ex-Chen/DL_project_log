def __init__(self, names, reg_pts=None, view_img=False, line_thickness=2,
    region_thickness=5, spdl_dist_thresh=10):
    """
        Initializes the SpeedEstimator with the given parameters.

        Args:
            names (dict): Dictionary of class names.
            reg_pts (list, optional): List of region points for speed estimation. Defaults to [(20, 400), (1260, 400)].
            view_img (bool, optional): Whether to display the image with annotations. Defaults to False.
            line_thickness (int, optional): Thickness of the lines for drawing boxes and tracks. Defaults to 2.
            region_thickness (int, optional): Thickness of the region lines. Defaults to 5.
            spdl_dist_thresh (int, optional): Distance threshold for speed calculation. Defaults to 10.
        """
    self.im0 = None
    self.annotator = None
    self.view_img = view_img
    self.reg_pts = reg_pts if reg_pts is not None else [(20, 400), (1260, 400)]
    self.region_thickness = region_thickness
    self.clss = None
    self.names = names
    self.boxes = None
    self.trk_ids = None
    self.trk_pts = None
    self.line_thickness = line_thickness
    self.trk_history = defaultdict(list)
    self.current_time = 0
    self.dist_data = {}
    self.trk_idslist = []
    self.spdl_dist_thresh = spdl_dist_thresh
    self.trk_previous_times = {}
    self.trk_previous_points = {}
    self.env_check = check_imshow(warn=True)
