def __init__(self, classes_names, imw=0, imh=0, colormap=cv2.COLORMAP_JET,
    heatmap_alpha=0.5, view_img=False, view_in_counts=True, view_out_counts
    =True, count_reg_pts=None, count_txt_color=(0, 0, 0), count_bg_color=(
    255, 255, 255), count_reg_color=(255, 0, 255), region_thickness=5,
    line_dist_thresh=15, line_thickness=2, decay_factor=0.99, shape='circle'):
    """Initializes the heatmap class with default values for Visual, Image, track, count and heatmap parameters."""
    self.annotator = None
    self.view_img = view_img
    self.shape = shape
    self.initialized = False
    self.names = classes_names
    self.imw = imw
    self.imh = imh
    self.im0 = None
    self.tf = line_thickness
    self.view_in_counts = view_in_counts
    self.view_out_counts = view_out_counts
    self.colormap = colormap
    self.heatmap = None
    self.heatmap_alpha = heatmap_alpha
    self.boxes = []
    self.track_ids = []
    self.clss = []
    self.track_history = defaultdict(list)
    self.counting_region = None
    self.line_dist_thresh = line_dist_thresh
    self.region_thickness = region_thickness
    self.region_color = count_reg_color
    self.in_counts = 0
    self.out_counts = 0
    self.count_ids = []
    self.class_wise_count = {}
    self.count_txt_color = count_txt_color
    self.count_bg_color = count_bg_color
    self.cls_txtdisplay_gap = 50
    self.decay_factor = decay_factor
    self.env_check = check_imshow(warn=True)
    self.count_reg_pts = count_reg_pts
    print(self.count_reg_pts)
    if self.count_reg_pts is not None:
        if len(self.count_reg_pts) == 2:
            print('Line Counter Initiated.')
            self.counting_region = LineString(self.count_reg_pts)
        elif len(self.count_reg_pts) >= 3:
            print('Polygon Counter Initiated.')
            self.counting_region = Polygon(self.count_reg_pts)
        else:
            print(
                'Invalid Region points provided, region_points must be 2 for lines or >= 3 for polygons.'
                )
            print('Using Line Counter Now')
            self.counting_region = LineString(self.count_reg_pts)
    if self.shape not in {'circle', 'rect'}:
        print("Unknown shape value provided, 'circle' & 'rect' supported")
        print('Using Circular shape now')
        self.shape = 'circle'
