def __init__(self, points: np.ndarray, scores: np.ndarray=None, data: Any=
    None, label: Hashable=None, embedding=None):
    self.points = validate_points(points)
    self.scores = scores
    self.data = data
    self.label = label
    self.absolute_points = self.points.copy()
    self.embedding = embedding
    self.age = None
