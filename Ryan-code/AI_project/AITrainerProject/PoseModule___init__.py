def __init__(self, mode=False, complexity=1, smoothLm=True, enable=False,
    smoothSe=True, detectionCon=0.5, trackCon=0.5):
    self.mode = mode
    self.complexity = complexity
    self.smoothLm = smoothLm
    self.enable = enable
    self.smoothSe = smoothSe
    self.detectionCon = detectionCon
    self.trackCon = trackCon
    self.mpDraw = mp.solutions.drawing_utils
    self.mpPose = mp.solutions.pose
    self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smoothLm,
        self.enable, self.smoothSe, self.detectionCon, self.trackCon)
