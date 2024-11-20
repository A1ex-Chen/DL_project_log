def __init__(self, classes_names, reg_pts=None, count_reg_color=(255, 0, 
    255), count_txt_color=(0, 0, 0), count_bg_color=(255, 255, 255),
    line_thickness=2, track_thickness=2, view_img=False, view_in_counts=
    True, view_out_counts=True, draw_tracks=False, track_color=None,
    region_thickness=5, line_dist_thresh=15, cls_txtdisplay_gap=50):
    """
        Initializes the ObjectCounter with various tracking and counting parameters.

        Args:
            classes_names (dict): Dictionary of class names.
            reg_pts (list): List of points defining the counting region.
            count_reg_color (tuple): RGB color of the counting region.
            count_txt_color (tuple): RGB color of the count text.
            count_bg_color (tuple): RGB color of the count text background.
            line_thickness (int): Line thickness for bounding boxes.
            track_thickness (int): Thickness of the track lines.
            view_img (bool): Flag to control whether to display the video stream.
            view_in_counts (bool): Flag to control whether to display the in counts on the video stream.
            view_out_counts (bool): Flag to control whether to display the out counts on the video stream.
            draw_tracks (bool): Flag to control whether to draw the object tracks.
            track_color (tuple): RGB color of the tracks.
            region_thickness (int): Thickness of the object counting region.
            line_dist_thresh (int): Euclidean distance threshold for line counter.
            cls_txtdisplay_gap (int): Display gap between each class count.
        """
    self.is_drawing = False
    self.selected_point = None
    self.reg_pts = [(20, 400), (1260, 400)] if reg_pts is None else reg_pts
    self.line_dist_thresh = line_dist_thresh
    self.counting_region = None
    self.region_color = count_reg_color
    self.region_thickness = region_thickness
    self.im0 = None
    self.tf = line_thickness
    self.view_img = view_img
    self.view_in_counts = view_in_counts
    self.view_out_counts = view_out_counts
    self.names = classes_names
    self.annotator = None
    self.window_name = 'Ultralytics YOLOv8 Object Counter'
    self.in_counts = 0
    self.out_counts = 0
    self.count_ids = []
    self.class_wise_count = {}
    self.count_txt_thickness = 0
    self.count_txt_color = count_txt_color
    self.count_bg_color = count_bg_color
    self.cls_txtdisplay_gap = cls_txtdisplay_gap
    self.fontsize = 0.6
    self.track_history = defaultdict(list)
    self.track_thickness = track_thickness
    self.draw_tracks = draw_tracks
    self.track_color = track_color
    self.env_check = check_imshow(warn=True)
    if len(self.reg_pts) == 2:
        print('Line Counter Initiated.')
        self.counting_region = LineString(self.reg_pts)
    elif len(self.reg_pts) >= 3:
        print('Polygon Counter Initiated.')
        self.counting_region = Polygon(self.reg_pts)
    else:
        print(
            'Invalid Region points provided, region_points must be 2 for lines or >= 3 for polygons.'
            )
        print('Using Line Counter Now')
        self.counting_region = LineString(self.reg_pts)
