def __init__(self, points, values, threshold=0.0):
    self.points = points
    self.values = values
    self.delaunay = Delaunay(self.points)
    self.threshold = threshold
