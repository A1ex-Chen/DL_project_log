def __init__(self, staticMode=False, maxFaces=2, refinelm=False,
    minDetectorCon=0.5, minTrackCon=0.5):
    self.staticMode = staticMode
    self.maxFaces = maxFaces
    self.refinelm = refinelm
    self.minDetectorCon = minDetectorCon
    self.minTrackCon = minTrackCon
    self.mpDraw = mp.solutions.drawing_utils
    self.mpFaceMash = mp.solutions.face_mesh
    self.faceMesh = self.mpFaceMash.FaceMesh(self.staticMode, self.maxFaces,
        self.refinelm, self.minDetectorCon, self.minTrackCon)
    self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
