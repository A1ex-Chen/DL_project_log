def __init__(self, names, pixels_per_meter=10, view_img=False,
    line_thickness=2, line_color=(255, 255, 0), centroid_color=(255, 0, 255)):
    """
        Initializes the DistanceCalculation class with the given parameters.

        Args:
            names (dict): Dictionary mapping class indices to class names.
            pixels_per_meter (int, optional): Conversion factor from pixels to meters. Defaults to 10.
            view_img (bool, optional): Flag to indicate if the video stream should be displayed. Defaults to False.
            line_thickness (int, optional): Thickness of the lines drawn on the image. Defaults to 2.
            line_color (tuple, optional): Color of the lines drawn on the image (BGR format). Defaults to (255, 255, 0).
            centroid_color (tuple, optional): Color of the centroids drawn (BGR format). Defaults to (255, 0, 255).
        """
    self.im0 = None
    self.annotator = None
    self.view_img = view_img
    self.line_color = line_color
    self.centroid_color = centroid_color
    self.clss = None
    self.names = names
    self.boxes = None
    self.line_thickness = line_thickness
    self.trk_ids = None
    self.centroids = []
    self.pixel_per_meter = pixels_per_meter
    self.left_mouse_count = 0
    self.selected_boxes = {}
    self.env_check = check_imshow(warn=True)
