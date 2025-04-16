def __init__(self, minDetectionCon=0.75):
    self.minDetectionCon = minDetectionCon
    self.mpFaceDetection = mp.solutions.face_detection
    self.mpDraw = mp.solutions.drawing_utils
    self.faceDetection = self.mpFaceDetection.FaceDetection(
        min_detection_confidence=self.minDetectionCon)
