def __init__(self, src=0):
    self.stream = cv2.VideoCapture(src)
    self.grabbed, self.frame = self.stream.read()
    self.stopped = False
