def __init__(self, kpts_to_check, line_thickness=2, view_img=False,
    pose_up_angle=145.0, pose_down_angle=90.0, pose_type='pullup'):
    """
        Initializes the AIGym class with the specified parameters.

        Args:
            kpts_to_check (list): Indices of keypoints to check.
            line_thickness (int, optional): Thickness of the lines drawn. Defaults to 2.
            view_img (bool, optional): Flag to display the image. Defaults to False.
            pose_up_angle (float, optional): Angle threshold for the 'up' pose. Defaults to 145.0.
            pose_down_angle (float, optional): Angle threshold for the 'down' pose. Defaults to 90.0.
            pose_type (str, optional): Type of pose to detect ('pullup', 'pushup', 'abworkout'). Defaults to "pullup".
        """
    self.im0 = None
    self.tf = line_thickness
    self.keypoints = None
    self.poseup_angle = pose_up_angle
    self.posedown_angle = pose_down_angle
    self.threshold = 0.001
    self.angle = None
    self.count = None
    self.stage = None
    self.pose_type = pose_type
    self.kpts_to_check = kpts_to_check
    self.view_img = view_img
    self.annotator = None
    self.env_check = check_imshow(warn=True)
    self.count = []
    self.angle = []
    self.stage = []
