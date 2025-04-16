def __init__(self, classID, confidence, x1, x2, y1, y2, image_width,
    image_height):
    self.classID = classID
    self.confidence = confidence
    self.x1 = x1
    self.x2 = x2
    self.y1 = y1
    self.y2 = y2
    self.u1 = x1 / image_width
    self.u2 = x2 / image_width
    self.v1 = y1 / image_height
    self.v2 = y2 / image_height
