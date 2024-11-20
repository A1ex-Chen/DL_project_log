def __init__(self, classes_names, reg_pts=None, line_thickness=2,
    track_thickness=2, view_img=False, region_color=(255, 0, 255),
    view_queue_counts=True, draw_tracks=False, count_txt_color=(255, 255, 
    255), track_color=None, region_thickness=5, fontsize=0.7):
    """
        Initializes the QueueManager with specified parameters for tracking and counting objects.

        Args:
            classes_names (dict): A dictionary mapping class IDs to class names.
            reg_pts (list of tuples, optional): Points defining the counting region polygon. Defaults to a predefined
                rectangle.
            line_thickness (int, optional): Thickness of the annotation lines. Defaults to 2.
            track_thickness (int, optional): Thickness of the track lines. Defaults to 2.
            view_img (bool, optional): Whether to display the image frames. Defaults to False.
            region_color (tuple, optional): Color of the counting region lines (BGR). Defaults to (255, 0, 255).
            view_queue_counts (bool, optional): Whether to display the queue counts. Defaults to True.
            draw_tracks (bool, optional): Whether to draw tracks of the objects. Defaults to False.
            count_txt_color (tuple, optional): Color of the count text (BGR). Defaults to (255, 255, 255).
            track_color (tuple, optional): Color of the tracks. If None, different colors will be used for different
                tracks. Defaults to None.
            region_thickness (int, optional): Thickness of the counting region lines. Defaults to 5.
            fontsize (float, optional): Font size for the text annotations. Defaults to 0.7.
        """
    self.is_drawing = False
    self.selected_point = None
    self.reg_pts = reg_pts if reg_pts is not None else [(20, 60), (20, 680),
        (1120, 680), (1120, 60)]
    self.counting_region = Polygon(self.reg_pts) if len(self.reg_pts
        ) >= 3 else Polygon([(20, 60), (20, 680), (1120, 680), (1120, 60)])
    self.region_color = region_color
    self.region_thickness = region_thickness
    self.im0 = None
    self.tf = line_thickness
    self.view_img = view_img
    self.view_queue_counts = view_queue_counts
    self.fontsize = fontsize
    self.names = classes_names
    self.annotator = None
    self.window_name = 'Ultralytics YOLOv8 Queue Manager'
    self.counts = 0
    self.count_txt_color = count_txt_color
    self.track_history = defaultdict(list)
    self.track_thickness = track_thickness
    self.draw_tracks = draw_tracks
    self.track_color = track_color
    self.env_check = check_imshow(warn=True)
