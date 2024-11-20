def update(self, points, values, reduce_to_active=True):
    if reduce_to_active:
        active_simplices = self.active_simplices()
        active_point_idx = np.unique(active_simplices.flatten())
        self.points = self.points[active_point_idx]
        self.values = self.values[active_point_idx]
    self.points = np.concatenate([self.points, points], axis=0)
    self.values = np.concatenate([self.values, values], axis=0)
    self.delaunay = Delaunay(self.points)
