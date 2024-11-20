def __call__(self, x, y):
    distance = np.linalg.norm(x - y)
    if distance < self.min_limit:
        return inf
    if distance > self.max_limit:
        return inf
    return distance
